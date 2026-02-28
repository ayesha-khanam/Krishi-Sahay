from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load agriculture data
with open("data/agriculture.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Generate embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "agri_index.faiss")

# Save documents
import pickle
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ FAISS Index Built Successfully!")