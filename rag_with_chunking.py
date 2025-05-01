import json
import requests
import os
import sys
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import numpy as np

# Print some diagnostics to help understand the environment
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

# === Locate PDF files in the project ===
project_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(project_dir, "KG-RAG-datasets", "nih-clinical-trial-protocols", "data", "v1", "docs")

# Verify the directory exists
if not os.path.exists(pdf_dir):
    print(f"Error: Could not find PDF directory at {pdf_dir}")
    # Search for PDFs elsewhere in the project
    found_pdfs = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.pdf'):
                found_pdfs.append(os.path.join(root, file))
    
    if found_pdfs:
        print(f"Found {len(found_pdfs)} PDFs in other locations.")
        pdf_dir = os.path.dirname(found_pdfs[0])
    else:
        print("No PDF files found in the project.")
        sys.exit(1)

pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith('.pdf')]
print(f"\nFound {len(pdf_files)} PDF files in {pdf_dir}")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    # Add page number marker for page-based chunking
                    text += f"[PAGE {page_num + 1}]\n{extracted}\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return f"Error processing document: {os.path.basename(pdf_path)}"

# === Extract text from PDFs ===
print("\nExtracting text from PDFs...")
document_texts = {}
for i, pdf_path in enumerate(pdf_files):
    filename = os.path.basename(pdf_path)
    print(f"Processing {i+1}/{len(pdf_files)}: {filename}")
    document_texts[filename] = extract_text_from_pdf(pdf_path)

# === CHUNKING STRATEGIES ===
print("\nApplying different chunking strategies...")

# Helper function to clean text
def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very short lines that might be page numbers or artifacts
    text = re.sub(r'^.{1,5}$', '', text, flags=re.MULTILINE)
    return text.strip()

# Strategy 1: Document-level chunks (baseline)
def chunk_by_document(document_texts):
    chunks = []
    for filename, text in document_texts.items():
        chunks.append({
            "text": clean_text(text),
            "source": filename,
            "strategy": "document"
        })
    return chunks

# Strategy 2: Page-level chunks
def chunk_by_page(document_texts):
    chunks = []
    for filename, text in document_texts.items():
        # Split by page markers we added during extraction
        pages = re.split(r'\[PAGE \d+\]\n', text)
        # Skip the first element if it's empty (due to the split)
        if pages and not pages[0].strip():
            pages = pages[1:]
        
        for i, page_text in enumerate(pages):
            if page_text.strip():
                chunks.append({
                    "text": clean_text(page_text),
                    "source": f"{filename}#page{i+1}",
                    "strategy": "page"
                })
    return chunks

# Strategy 3: Fixed-size chunks with overlap
def chunk_by_fixed_size(document_texts, chunk_size=1000, overlap=200):
    chunks = []
    for filename, text in document_texts.items():
        clean = clean_text(text)
        
        # Create chunks of approximately chunk_size characters
        if len(clean) <= chunk_size:
            chunks.append({
                "text": clean,
                "source": f"{filename}#fixed1",
                "strategy": "fixed-size"
            })
        else:
            for i in range(0, len(clean), chunk_size - overlap):
                chunk_text = clean[i:i + chunk_size]
                if len(chunk_text) >= 100:  # Avoid very small chunks at the end
                    chunks.append({
                        "text": chunk_text,
                        "source": f"{filename}#fixed{i//chunk_size + 1}",
                        "strategy": "fixed-size"
                    })
    return chunks

# Strategy 4: Paragraph-based chunks
def chunk_by_paragraph(document_texts, min_length=100):
    chunks = []
    for filename, text in document_texts.items():
        # Split by double newlines to identify paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_count = 1
        
        for para in paragraphs:
            clean_para = clean_text(para)
            
            # Skip empty paragraphs
            if not clean_para:
                continue
                
            # If the paragraph is very short, combine with next one
            if len(clean_para) < min_length:
                current_chunk += " " + clean_para
            else:
                # If we have accumulated text, create a chunk
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "source": f"{filename}#para{chunk_count}",
                        "strategy": "paragraph"
                    })
                    chunk_count += 1
                    current_chunk = ""
                
                # Add the current paragraph as its own chunk
                chunks.append({
                    "text": clean_para,
                    "source": f"{filename}#para{chunk_count}",
                    "strategy": "paragraph"
                })
                chunk_count += 1
        
        # Don't forget any remaining accumulated text
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": f"{filename}#para{chunk_count}",
                "strategy": "paragraph"
            })
    
    return chunks

# Strategy 5: Section-based chunks (using headers)
def chunk_by_section(document_texts):
    chunks = []
    
    # Common headers in clinical trial protocols
    section_headers = [
        r'INTRODUCTION', r'BACKGROUND', r'OBJECTIVE', r'METHODS', r'METHODOLOGY', 
        r'MATERIALS AND METHODS', r'RESULTS', r'DISCUSSION', r'CONCLUSION', r'REFERENCES',
        r'ELIGIBILITY CRITERIA', r'INCLUSION CRITERIA', r'EXCLUSION CRITERIA',
        r'STUDY DESIGN', r'STATISTICAL ANALYSIS', r'ADVERSE EVENTS', r'PROTOCOL',
        r'SUMMARY', r'ABSTRACT', r'PURPOSE', r'AIMS', r'ENDPOINTS', r'PROCEDURES',
        r'INTERVENTION', r'TREATMENT', r'FOLLOW-UP', r'ASSESSMENT'
    ]
    
    # Create a regex pattern for section headers
    header_pattern = r'(?i)(?:^|\n)(?:\d+[\.\s]+)*(' + '|'.join(section_headers) + r')(?::|\s*\n|\s*$)'
    
    for filename, text in document_texts.items():
        # Find all potential section headers
        matches = list(re.finditer(header_pattern, text))
        
        if not matches:
            # If no sections found, treat the entire document as one section
            chunks.append({
                "text": clean_text(text),
                "source": f"{filename}#fulltext",
                "strategy": "section"
            })
            continue
        
        # Create chunks based on sections
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Determine the end position (either the next section or the end of text)
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            section_text = text[start_pos:end_pos]
            section_name = match.group(1).strip()
            
            if len(section_text) > 50:  # Only include non-trivial sections
                chunks.append({
                    "text": clean_text(section_text),
                    "source": f"{filename}#{section_name.lower()}",
                    "strategy": "section"
                })
    
    return chunks

# Apply all chunking strategies
doc_chunks = chunk_by_document(document_texts)
page_chunks = chunk_by_page(document_texts)
fixed_chunks = chunk_by_fixed_size(document_texts)
para_chunks = chunk_by_paragraph(document_texts)
section_chunks = chunk_by_section(document_texts)

# Combine all chunks and assign unique IDs
all_chunks = []
chunk_id = 0

for chunk_list in [doc_chunks, page_chunks, fixed_chunks, para_chunks, section_chunks]:
    for chunk in chunk_list:
        chunk["id"] = chunk_id
        all_chunks.append(chunk)
        chunk_id += 1

# Print statistics about chunks
strategies = ["document", "page", "fixed-size", "paragraph", "section"]
print("\nChunking statistics:")
for strategy in strategies:
    strategy_chunks = [c for c in all_chunks if c["strategy"] == strategy]
    if strategy_chunks:
        lengths = [len(c["text"]) for c in strategy_chunks]
        print(f"  {strategy}: {len(strategy_chunks)} chunks, avg length: {np.mean(lengths):.1f} chars")

# === Create or load sample questions ===
sample_questions = [
    {
        "question": "What are the eligibility criteria for the clinical trial?",
        "answer": "eligibility criteria",
        "type": "semantic",
        "expected_strategy": "section"
    },
    {
        "question": "What are the adverse effects reported in the study?",
        "answer": "adverse effects",
        "type": "semantic",
        "expected_strategy": "section"
    },
    {
        "question": "What is the dosage of the medication used?",
        "answer": "dosage",
        "type": "keyword",
        "expected_strategy": "paragraph"
    },
    {
        "question": "What was the primary outcome measure?",
        "answer": "primary outcome",
        "type": "semantic",
        "expected_strategy": "section"
    },
    {
        "question": "What were the exclusion criteria?",
        "answer": "exclusion criteria",
        "type": "semantic",
        "expected_strategy": "section"
    },
    {
        "question": "How many patients were enrolled in the study?",
        "answer": "patients enrolled",
        "type": "keyword",
        "expected_strategy": "paragraph"
    },
    {
        "question": "What is the purpose of this clinical trial?",
        "answer": "purpose",
        "type": "semantic",
        "expected_strategy": "document"
    },
    {
        "question": "What are the potential risks of participation?",
        "answer": "risks",
        "type": "semantic",
        "expected_strategy": "section"
    }
]

# Try to load questions from GitHub, but use sample ones if there's an error
try:
    questions_url = "https://raw.githubusercontent.com/docugami/KG-RAG-datasets/main/NIH-Clinical-Trials/questions.json"
    response = requests.get(questions_url)
    if response.status_code == 200:
        try:
            github_questions = json.loads(response.text)
            if isinstance(github_questions, list):
                print("\nLoaded questions from GitHub")
                # Extend with our custom questions
                github_questions.extend(sample_questions)
                questions = github_questions
            else:
                questions = sample_questions
        except json.JSONDecodeError:
            print("\nUsing sample questions (JSON parse error)")
            questions = sample_questions
    else:
        print(f"\nUsing sample questions (HTTP {response.status_code})")
        questions = sample_questions
except Exception as e:
    print(f"\nUsing sample questions ({str(e)})")
    questions = sample_questions

# === Generate Embeddings ===
print("\nInitializing sentence transformer model...")
model = SentenceTransformer("all-mpnet-base-v2")  # Using the better model based on previous results

# Extract texts from all chunks
texts = [chunk["text"] for chunk in all_chunks]
print(f"Encoding {len(texts)} chunks...")

# Encode in batches to avoid memory issues
batch_size = 32
embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    print(f"Encoding batch {i//batch_size + 1}/{len(texts)//batch_size + 1}...")
    batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings)
print(f"Generated {len(embeddings)} embeddings of size {embeddings[0].shape[0]}")

# === Connect to Qdrant and Upload ===
print("\nConnecting to Qdrant...")
try:
    client = QdrantClient("localhost", port=6333)
    # Test connection
    client.get_collections()
    print("Successfully connected to Qdrant")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Using in-memory Qdrant instance instead")
    client = QdrantClient(":memory:")

# Create collection for chunking strategies
collection_name = "clinical_trial_chunks"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

print(f"Creating collection: {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings[0].shape[0], distance=Distance.COSINE)
)

# Create points with payload including chunking strategy
points = []
for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "strategy": chunk["strategy"]
            }
        )
    )

print(f"Uploading {len(points)} vectors...")
client.upload_points(collection_name, points=points)

# === Evaluate Retrieval by Chunking Strategy ===
def evaluate_retrieval(questions, collection_name, limit=5):
    results = {strategy: {"correct": 0, "total": 0} for strategy in strategies}
    question_results = []
    
    print(f"\nEvaluating retrieval with {limit} results per query...")
    
    for q in questions:
        query_text = q["question"]
        expected_answer = q["answer"].lower()
        
        # Get query vector
        query_vector = model.encode(query_text)
        
        # Search for similar chunks
        search_results = client.search(collection_name, query_vector.tolist(), limit=limit)
        
        # Process results
        retrieved_chunks = []
        found_answer = False
        best_strategy = None
        best_score = 0
        
        for result in search_results:
            strategy = result.payload["strategy"]
            text = result.payload["text"].lower()
            source = result.payload["source"]
            score = result.score
            
            # Check if this chunk contains the answer
            contains_answer = expected_answer in text
            
            # If this is the first chunk with the answer, note the strategy
            if contains_answer and not found_answer:
                found_answer = True
                best_strategy = strategy
                best_score = score
            
            # Store details about this result
            retrieved_chunks.append({
                "strategy": strategy,
                "source": source,
                "score": score,
                "contains_answer": contains_answer,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            })
        
        # If we found the answer, count it as correct for that strategy
        if found_answer:
            results[best_strategy]["correct"] += 1
        
        # Count this question for the expected strategy
        if "expected_strategy" in q:
            expected_strategy = q["expected_strategy"]
            results[expected_strategy]["total"] += 1
        
        # Store details about this question's results
        question_results.append({
            "question": query_text,
            "answer_found": found_answer,
            "best_strategy": best_strategy if found_answer else None,
            "best_score": best_score if found_answer else None,
            "retrieved_chunks": retrieved_chunks
        })
    
    # Calculate accuracy for each strategy
    for strategy in strategies:
        if results[strategy]["total"] > 0:
            results[strategy]["accuracy"] = results[strategy]["correct"] / results[strategy]["total"]
        else:
            results[strategy]["accuracy"] = 0
    
    return results, question_results

# Run evaluation
strategy_results, question_results = evaluate_retrieval(questions, collection_name)

# === Visualize Results ===
# Plot accuracy by chunking strategy
strategies_with_data = [s for s in strategies if strategy_results[s]["total"] > 0]
accuracies = [strategy_results[s]["accuracy"] for s in strategies_with_data]

plt.figure(figsize=(10, 6))
bars = plt.bar(strategies_with_data, accuracies, color="#55A868")

plt.title("Retrieval Accuracy by Chunking Strategy")
plt.xlabel("Chunking Strategy")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("chunking_comparison.png")
print("\nSaved visualization to chunking_comparison.png")

# Create a second visualization: breakdown of strategies used in top results
strategy_counts = {s: 0 for s in strategies}
for qr in question_results:
    # Count the strategies in the top results for each question
    for chunk in qr["retrieved_chunks"]:
        strategy_counts[chunk["strategy"]] += 1

# Normalize by total number of results
total_results = sum(strategy_counts.values())
normalized_counts = {s: count/total_results for s, count in strategy_counts.items()}

plt.figure(figsize=(10, 6))
bars = plt.bar(strategies, [normalized_counts[s] for s in strategies], color="#4C72B0")

plt.title("Distribution of Chunking Strategies in Top Results")
plt.xlabel("Chunking Strategy")
plt.ylabel("Proportion of Results")
plt.ylim(0, max(normalized_counts.values()) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("strategy_distribution.png")
print("Saved visualization to strategy_distribution.png")

# === Print Detailed Results ===
print("\n=== Evaluation Results ===")
print("\nAccuracy by Chunking Strategy:")
for strategy in strategies:
    correct = strategy_results[strategy]["correct"]
    total = strategy_results[strategy]["total"]
    accuracy = strategy_results[strategy]["accuracy"] if total > 0 else "N/A"
    print(f"  {strategy}: {correct}/{total} = {accuracy:.2f}" if total > 0 else f"  {strategy}: {correct}/{total} = {accuracy}")

print("\nStrategy Distribution in Top Results:")
for strategy in strategies:
    count = strategy_counts[strategy]
    percent = normalized_counts[strategy] * 100
    print(f"  {strategy}: {count} results ({percent:.1f}%)")

print("\n=== Sample Query Results ===")
for i, qr in enumerate(question_results[:3]):  # Show first 3 questions
    print(f"\nQ{i+1}: {qr['question']}")
    print(f"Answer found: {qr['answer_found']}")
    if qr['answer_found']:
        print(f"Best strategy: {qr['best_strategy']} (score: {qr['best_score']:.4f})")
    
    print("Top chunks:")
    for j, chunk in enumerate(qr["retrieved_chunks"][:3]):  # Show top 3 chunks
        print(f"  {j+1}. [{chunk['strategy']}] {chunk['source']} (score: {chunk['score']:.4f})")
        print(f"     Contains answer: {chunk['contains_answer']}")
        print(f"     Preview: {chunk['text_preview']}")

print("\nCheck chunking_comparison.png and strategy_distribution.png for visualizations")