import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Environment Variables
load_dotenv()

DATA_PATH = "data/policies/"
DB_PATH = "data/chroma_db_multi"

# --- 1. CLEAN START ---
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH) 
    print("🧹 Old database cleared for fresh semantic indexing.")

def ingest_structured():
    # Load all PDFs from the directory
    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    
    valid_states = ["Tennessee", "Washington", "California", "Texas", "New York"]
    section_docs = []

    # --- STEP 1: INITIAL SECTIONAL SPLIT ---
    # We first split by the physical word "Section" found in your PDFs
    for doc in raw_docs:
        filename = doc.metadata.get("source", "Unknown")
        state = next((s for s in valid_states if s in filename), "N/A")
        year = 2024 if "2024" in filename else (2022 if "2022" in filename else 2023)

        content = doc.page_content
        # Split by "Section" and filter out empty strings
        parts = [p.strip() for p in content.split("Section") if p.strip()]
        
        for part in parts:
            # Re-format to keep the "Section" context for the LLM
            full_text = f"Section {part}" if part[0].isdigit() else part
            
            section_docs.append(Document(
                page_content=full_text,
                metadata={"state": state, "year": year, "source": filename}
            ))

    # --- STEP 2: SEMANTIC RECURSIVE CHUNKING ---
    # Now we break those large Sections into smaller 500-char pieces.
    # This ensures "PTO" and "Internet" stay in different chunks!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # Priority: Paragraphs -> New Lines -> Policy Sections -> Bullets -> Sentences
        separators=["\n\n", "\n", "Section ", "●", "•", ". ", " ", ""],
        add_start_index=True
    )

    # This creates the final atomic units for the Vector Store
    semantic_chunks = text_splitter.split_documents(section_docs)

    # --- STEP 3: VECTOR STORE INITIALIZATION ---
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=semantic_chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"🚀 SUCCESS: Ingested {len(semantic_chunks)} semantic chunks.")
    print(f"📊 Audit: Created {len(section_docs)} parent sections.")

if __name__ == "__main__":
    ingest_structured()