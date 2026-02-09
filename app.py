import gradio as gr
import os
import pandas as pd
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation

# 1. Initialize RAG System 
# Make sure the path matches your 'data' folder structure on GitHub
try:
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain, retriever = None, None

def predict(question):
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ Error: OpenAI API Key not found!", pd.DataFrame()

    try:
        # 1. Generate the raw answer
        raw_answer = rag_chain.invoke(question)
        
        # 2. Retrieve contexts for evaluation
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]
        ground_truth = "Refer to official policy."
        
        # 3. Run RAGAS Evaluation
        report = run_evaluation(question, raw_answer, contexts, ground_truth)
        faithfulness_score = report['faithfulness'].iloc[0]
        
        # 4. SAFETY GUARDRAIL: Check if the AI stayed on topic
        # If score is below 0.4, the AI likely hallucinated or was hijacked
        if faithfulness_score < 0.4:
            final_answer = (
                "🛡️ [SECURITY BLOCK]: This response was censored because it failed "
                "the Faithfulness check. The AI attempted to deviate from the policy document."
            )
        else:
            final_answer = raw_answer
            
        # Prepare metrics table
        metrics_data = {
            "Metric": ["Faithfulness", "Answer Relevancy", "Answer Correctness"],
            "Score": [
                round(faithfulness_score, 2),
                round(report['answer_relevancy'].iloc[0], 2),
                round(report['answer_correctness'].iloc[0], 2)
            ]
        }
        return final_answer, pd.DataFrame(metrics_data)

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()