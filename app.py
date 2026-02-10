import gradio as gr
import os
import pandas as pd
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation

# 1. Initialize RAG System 
try:
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain, retriever = None, None

def predict(question):
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ Error: OpenAI API Key not found in Space Secrets!", pd.DataFrame()

    if rag_chain is None:
        return "⚠️ Error: RAG system failed to initialize. Check PDF path.", pd.DataFrame()

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
        
        # 4. SAFETY GUARDRAIL
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

# --- THE MISSING UI SECTION ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    gr.Markdown("RAG System with automated **RAGAS** safety guardrails.")
    
    with gr.Tab("💬 Chat"):
        input_text = gr.Textbox(label="Your Question", placeholder="Ask a policy question...")
        output_text = gr.Textbox(label="AI Response", interactive=False)
        btn = gr.Button("Analyze Policy", variant="primary")

    with gr.Tab("📊 Quality Metrics"):
        output_table = gr.DataFrame(label="RAGAS Evaluation Scores")

    btn.click(fn=predict, inputs=input_text, outputs=[output_text, output_table])

# --- THE MISSING LAUNCH SECTION ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )