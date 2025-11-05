from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import os

# Load your documents (replace path with yours)
loader = DirectoryLoader("modules", glob="**/*.txt")
docs = loader.load()

# Split docs
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
db = FAISS.from_documents(chunks, embeddings)

# Save it
os.makedirs("faiss_index", exist_ok=True)
db.save_local("faiss_index")

print("✅ FAISS index created and saved successfully.")
