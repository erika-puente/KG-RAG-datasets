import json
import requests
import os
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader  # Updated import for newer PyPDF2 versions

# Print some diagnostics to help understand the environment
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

# === Load Clinical Trial Data from Local Files ===
# Detect the path structure in the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# List all files in the project directory to find potential PDF paths
print("\nSearching for PDF files in project directory...")
found_pdfs = []

for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith('.pdf'):
            found_pdfs.append(os.path.join(root, file))

if not found_pdfs:
    print("No PDF files found in the project directory.")
    
    # Try to use alternative approach - check if we can find the specific directory
    # based on the VS Code file structure in the screenshot
    potential_path = os.path.join(project_dir, "KG-RAG-datasets", "nih-clinical-trial-protocols", "data", "v1", "docs")
    if os.path.exists(potential_path):
        data_dir = potential_path
    else:
        print(f"Could not find: {potential_path}")
        
        # Try another path that might match the structure shown in the screenshot
        potential_path = os.path.join(project_dir, "RAG_FINAL_PROJECT", "KG-RAG-datasets", "data", "v1", "docs")
        if os.path.exists(potential_path):
            data_dir = potential_path
        else:
            print(f"Could not find: {potential_path}")
            
            # One more attempt with simplified path
            potential_path = os.path.join(project_dir, "data", "v1", "docs")
            if os.path.exists(potential_path):
                data_dir = potential_path
            else:
                print(f"Could not find: {potential_path}")
                
                # Let's check if the "docs" folder exists anywhere in the project
                docs_paths = []
                for root, dirs, files in os.walk(project_dir):
                    if "docs" in dirs:
                        docs_paths.append(os.path.join(root, "docs"))
                
                if docs_paths:
                    print(f"Found the following 'docs' directories:")
                    for i, path in enumerate(docs_paths):
                        print(f"{i+1}. {path}")
                    
                    # Use the first docs path found
                    data_dir = docs_paths[0]
                    print(f"Using the first docs path: {data_dir}")
                else:
                    print("No 'docs' directories found.")
                    print("Please provide the full path to the directory containing your PDF files:")
                    data_dir = input("> ")
else:
    # Use the directory of the first PDF we found
    pdf_dir = os.path.dirname(found_pdfs[0])
    print(f"Found PDF files in: {pdf_dir}")
    data_dir = pdf_dir

# Verify the directory exists and contains PDFs
if not os.path.exists(data_dir):
    print(f"Error: The directory {data_dir} does not exist.")
    sys.exit(1)

pdf_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.pdf')]
if not pdf_files:
    print(f"No PDF files found in {data_dir}")
    
    # If we didn't find PDFs in data_dir but found PDFs elsewhere, use those
    if found_pdfs:
        print(f"Using {len(found_pdfs)} PDFs found in other directories.")
        pdf_files = found_pdfs
    else:
        sys.exit(1)

print(f"\nFound {len(pdf_files)} PDF files:")
for pdf in pdf_files[:5]:  # Show the first 5 PDFs
    print(f"  - {os.path.basename(pdf)}")
if len(pdf_files) > 5:
    print(f"  - ... and {len(pdf_files) - 5} more")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return f"Error processing document: {os.path.basename(pdf_path)}"

# Extract text from each PDF file
print("\nExtracting text from PDFs...")
chunks = []
for i, pdf_path in enumerate(pdf_files):
    filename = os.path.basename(pdf_path)
    print(f"Processing {i+1}/{len(pdf_files)}: {filename}")
    text = extract_text_from_pdf(pdf_path)
    
    # Create chunks (here simply using the whole document as one chunk)
    chunks.append({
        "id": i,
        "text": text,
        "source": filename
    })

# === Create sample questions instead of fetching from GitHub ===
# Since we're having issues with the GitHub JSON, let's create our own sample questions
print("\nCreating sample questions...")
questions = [
    {
        "question": "What are the eligibility criteria for the clinical trial?",
        "answer": "eligibility criteria",
        "type": "single-doc-multi-chunk"
    },
    {
        "question": "What are the adverse effects reported in the study?",
        "answer": "adverse effects",
        "type": "multi-doc"
    },
    {
        "question": "What is the dosage of the medication used?",
        "answer": "dosage",
        "type": "single-doc-single-chunk"
    },
    {
        "question": "What was the primary outcome measure?",
        "answer": "primary outcome",
        "type": "single-doc-single-chunk"
    },
    {
        "question": "What were the exclusion criteria?",
        "answer": "exclusion criteria",
        "type": "single-doc-multi-chunk"
    }
]

# Try to fetch questions from GitHub if possible, but use our fallback if there's an error
try:
    questions_url = "https://raw.githubusercontent.com/docugami/KG-RAG-datasets/main/NIH-Clinical-Trials/questions.json"
    response = requests.get(questions_url)
    if response.status_code == 200:
        # Check if the content is valid JSON
        try:
            github_questions = json.loads(response.text)
            if isinstance(github_questions, list):
                print("Successfully loaded questions from GitHub")
                questions = github_questions
        except json.JSONDecodeError:
            print("Error parsing JSON from GitHub, using fallback questions")
    else:
        print(f"Error fetching questions: {response.status_code}, using fallback questions")
except Exception as e:
    print(f"Exception when fetching questions: {e}, using fallback questions")

texts = [chunk["text"] for chunk in chunks]
print(f"\nLoaded {len(texts)} text chunks")

# === Generate Embeddings ===
print("\nInitializing sentence transformer models...")
model_minilm = SentenceTransformer("all-MiniLM-L6-v2")
model_mpnet = SentenceTransformer("all-mpnet-base-v2")

print("Encoding documents...")
embeddings_minilm = model_minilm.encode(texts, show_progress_bar=True)
embeddings_mpnet = model_mpnet.encode(texts, show_progress_bar=True)

# === Connect to Qdrant and Upload ===
print("\nConnecting to Qdrant...")
try:
    client = QdrantClient("localhost", port=6333)
    # Test if Qdrant is available
    client.get_collections()
    print("Successfully connected to Qdrant")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Using in-memory Qdrant instance instead")
    client = QdrantClient(":memory:")

print("Creating Qdrant collections...")
client.recreate_collection("minilm_collection", VectorParams(size=384, distance=Distance.COSINE))
client.recreate_collection("mpnet_collection", VectorParams(size=768, distance=Distance.COSINE))

minilm_points = [PointStruct(id=i, vector=vec.tolist(), payload={"text": text}) for i, (vec, text) in enumerate(zip(embeddings_minilm, texts))]
mpnet_points = [PointStruct(id=i, vector=vec.tolist(), payload={"text": text}) for i, (vec, text) in enumerate(zip(embeddings_mpnet, texts))]

print("Uploading vectors...")
client.upload_points("minilm_collection", points=minilm_points)
client.upload_points("mpnet_collection", points=mpnet_points)

# === Evaluate Top-5 Accuracy ===
def evaluate_model(model, collection_name):
    correct = {"single-doc-single-chunk": 0, "single-doc-multi-chunk": 0, "multi-doc": 0}
    total = {"single-doc-single-chunk": 0, "single-doc-multi-chunk": 0, "multi-doc": 0}

    print(f"Evaluating {collection_name}...")
    for q in questions:
        q_type = q["type"]
        question_text = q["question"]
        expected_answer = q["answer"].lower()
        query_vector = model.encode(question_text)

        results = client.search(collection_name=collection_name, query_vector=query_vector.tolist(), limit=5)
        top_texts = [r.payload["text"].lower() for r in results]

        total[q_type] += 1
        if any(expected_answer in t for t in top_texts):
            correct[q_type] += 1

    return {k: correct[k] / total[k] if total[k] > 0 else 0 for k in total}

acc_minilm = evaluate_model(model_minilm, "minilm_collection")
acc_mpnet = evaluate_model(model_mpnet, "mpnet_collection")

# === Visualization ===
question_types = list(acc_minilm.keys())
index = range(len(question_types))
bar_width = 0.35

minilm_vals = [acc_minilm[q] for q in question_types]
mpnet_vals = [acc_mpnet[q] for q in question_types]

fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar([i - bar_width/2 for i in index], minilm_vals, bar_width, label="MiniLM", color="#4C72B0")
bar2 = ax.bar([i + bar_width/2 for i in index], mpnet_vals, bar_width, label="MPNet", color="#55A868")

ax.set_ylabel("Top-5 Accuracy")
ax.set_title("RAG Accuracy on Clinical Trial Questions by Type")
ax.set_xticks(index)
ax.set_xticklabels(question_types, rotation=15)
ax.legend()
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("rag_accuracy.png")
print("\nSaved visualization to rag_accuracy.png")

# === Print Results ===
print("\n✅ Accuracy by Model:")
print("MiniLM:", acc_minilm)
print("MPNet:", acc_mpnet)