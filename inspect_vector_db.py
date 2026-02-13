import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd

load_dotenv()
DB_PATH = "data/chroma_db_multi"

def inspect_sections():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    
    def identify_section(text):
        if "Section 3:" in text: return "Section 3"
        if "Section 2:" in text: return "Section 2"
        if "Section 1:" in text: return "Section 1"
        return "Other"

    df = pd.DataFrame({
        "State": [m.get("state", "N/A") for m in data["metadatas"]],
        "Year": [m.get("year", "N/A") for m in data["metadatas"]],
        "Section": [identify_section(doc) for doc in data["documents"]],
        "Text Preview": [doc[:150].replace('\n', ' ') + "..." for doc in data["documents"]]
    })
    
    # Sort so you see Section 1, 2, 3 for the SAME document together
    df = df.sort_values(by=["State", "Year", "Section"])
    
    print(f"\n📊 Total Chunks Found: {len(df)}")
    
    # Let's specifically look for Tennessee 2024 to verify the challenge
    tn_2024 = df[(df['State'] == 'Tennessee') & (df['Year'] == 2024)]
    print("\n🔍 Verification: Tennessee 2024 Sections")
    print(tn_2024.to_string(index=False))

if __name__ == "__main__":
    inspect_sections()