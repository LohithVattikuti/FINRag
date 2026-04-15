# Working with file and directory paths
import os
# Reading and writing JSON files
import json
# Progress bar for loops
from tqdm import tqdm

# Sentence-transformers model for text embeddings
from sentence_transformers import SentenceTransformer
# Qdrant client for interacting with the vector database
from qdrant_client import QdrantClient
# Data models for defining vector collection and points
from qdrant_client.http.models import VectorParams, PointStruct


# ─── Configuration ────────────────────────────────────────────────────────────
# Folder containing JSON chunks to embed

# Path to the folder with parsed text chunks
INPUT_DIR = "sec_chunks"

# Name of the Qdrant collection for storing embeddings
COLLECTION_NAME = "finrag_chunks"

# Pre-trained model used to generate embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Initialize embedding model ───────────────────────────────────────────────
# Load the embedding model into memory (downloads if not present)
model = SentenceTransformer(EMBEDDING_MODEL)

# ─── Connect to Qdrant ──────────────────────────────────────────────────────────
# Create a client to interact with local Qdrant service
qdrant = QdrantClient(url="http://localhost:6333")

# Ensure the collection is fresh by deleting if it exists, then creating it
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance="Cosine"
    )
)

# ─── Chunk Loader Function ────────────────────────────────────────────────────
# Generator to read each JSON chunk file and yield its ID, text, and metadata
def load_chunks(input_dir):
    # Loop through every file in the input directory
    for filename in os.listdir(input_dir):
        # Skip files that are not JSON chunks
        if not filename.endswith(".json"):
            continue

        # Build the full path to the JSON file
        path = os.path.join(input_dir, filename)
        # Load JSON data into a Python dictionary
        data = json.load(open(path, "r", encoding="utf-8"))

        # Construct a unique ID from metadata
        chunk_id = (
            f"{data['ticker']}_"
            f"{data['form_type']}_"
            f"{data['filing_date']}_"
            f"chunk{data['chunk_id']}"
        )

        # Extract the text content for embedding
        text = data["text"]
        # Prepare metadata for storing alongside the vector
        metadata = {
            "ticker": data["ticker"],
            "form_type": data["form_type"],
            "filing_date": data["filing_date"],
            "chunk_id": data["chunk_id"]
        }

        # Yield the ID, text, and metadata for embedding ingestion
        yield chunk_id, text, metadata

# ─── Embed & Upsert to Qdrant ───────────────────────────────────────────────
def main():
    """
    Main routine to:
      1. Read each chunk via load_chunks()
      2. Generate its embedding
      3. Upsert batches of embeddings + metadata into Qdrant
    """
    points = []
    for idx, (chunk_id, text, metadata) in enumerate(tqdm(load_chunks(INPUT_DIR), desc="Embedding & upserting")):
        # Store the original chunk identifier in metadata for traceability
        metadata["source_chunk_id"] = chunk_id
        vector = model.encode(text).tolist()
        point = PointStruct(id=idx, vector=vector, payload=metadata)
        points.append(point)
        if len(points) >= 100:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

if __name__ == "__main__":
    main()