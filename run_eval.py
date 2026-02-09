import os
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation
import pandas as pd


print("🚀 STARTING SCRIPT...")

def main():
    pdf_path = "data/company_policy.pdf" 
    
    # Check if file exists before doing anything
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: File not found at {pdf_path}")
        return

    print("--- 1. Initializing RAG System ---")
    try:
        # If it hangs here, the problem is in PDF loading or ChromaDB
        rag_chain, retriever = initialize_rag(pdf_path)
        print("✅ System Initialized Successfully")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    query = "What is the internet speed requirement for remote work?"
    ground_truth = "The internet speed requirement for remote work is a minimum of 50 Mbps."

    print(f"\n--- 2. Testing Query: {query} ---")
    try:
        print("⏳ Talking to OpenAI (Answer Generation)...")
        answer = rag_chain.invoke(query)
        print(f"🤖 AI Answer: {answer}")

        print("⏳ Talking to OpenAI (Evaluation)...")
        # If it hangs here, the problem is in the RAGAS evaluator logic
        docs = retriever.invoke(query)
        contexts = [d.page_content for d in docs]
        
        report = run_evaluation(query, answer, contexts, ground_truth)
        print("✅ Evaluation Complete")
        
        print("\n📈 FULL QUALITY DASHBOARD:")
        print(report.to_string())
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    print("🎯 Calling main()...")
    main()
    print("🏁 Script Finished.")