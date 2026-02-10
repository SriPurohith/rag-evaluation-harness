import gradio as gr
import os
import pandas as pd
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# Initialize RAG
try:
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"Error: {e}")
    rag_chain, retriever = None, None

import subprocess

def run_deepeval_audit():
    """Triggers the DeepEval CLI and captures the terminal output."""
    try:
        # We use 'python -m deepeval' for better compatibility in container environments
        result = subprocess.run(
            ["python", "-m", "deepeval", "test", "run", "test_deepeval.py"],
            capture_output=True,
            text=True,
            timeout=120 # Prevents the UI from hanging if the audit takes too long
        )
        # Return stdout if successful, otherwise stderr
        return result.stdout if result.returncode == 0 else f"⚠️ Audit Error:\n{result.stderr}"
    except Exception as e:
        return f"❌ Failed to trigger audit: {str(e)}"
    
def predict(question):
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ Key Missing", pd.DataFrame()

    try:
        # 1. Generate & Retrieve
        raw_answer = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

        # 2. DeepEval Guardrail (Explainable Security)
        # Threshold 0.5: Anything less than 50% grounded is blocked
        hallucination_metric = HallucinationMetric(threshold=0.5)
        test_case = LLMTestCase(
                    input=question,
                    actual_output=raw_answer,
                    context=contexts,
                    retrieval_context=contexts
                )
        hallucination_metric.measure(test_case)
        
        # 3. RAGAS Metrics (Statistical Quality)
        ragas_report = run_evaluation(question, raw_answer, contexts, "Refer to policy.")

        # 4. Decision Logic
        if hallucination_metric.score < 0.5:
            final_answer = (
                f"🛡️ [SECURITY BLOCK]: DeepEval flagged this response.\n\n"
                f"**Reason:** {hallucination_metric.reason}"
            )
        else:
            final_answer = raw_answer

        # 5. Combined Metrics Table
        metrics_data = {
            "Engine": ["DeepEval", "RAGAS", "RAGAS"],
            "Metric": ["Hallucination Score", "Faithfulness", "Answer Relevancy"],
            "Score": [
                round(hallucination_metric.score, 2),
                round(ragas_report['faithfulness'].iloc[0], 2),
                round(ragas_report['answer_relevancy'].iloc[0], 2)
            ],
            "Status": [
                "✅ Pass" if hallucination_metric.score >= 0.5 else "❌ Fail",
                "✅ Pass" if ragas_report['faithfulness'].iloc[0] >= 0.4 else "⚠️ Low",
                "N/A"
            ]
        }
        return final_answer, pd.DataFrame(metrics_data)

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

# 2. UI Layout (Gradio 6.5.1 Optimized)

with gr.Blocks(title="Policy-QA Eval Harness", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    
    with gr.Tab("💬 Secure Chat"):
        # You MUST have code indented here!
        input_box = gr.Textbox(label="Ask a Policy Question")
        output_text = gr.Markdown(label="Response")
        chat_btn = gr.Button("Submit")
        # Link the button to your predict function
        chat_btn.click(
            fn=predict, 
            inputs=input_box, 
            outputs=[output_text, metrics_table]
        )
    
    with gr.Tab("📊 Quality Engineering"):
        # Code for your RAGAS table goes here
        gr.Markdown("### 📈 RAGAS Metrics")
        metrics_table = gr.DataFrame() 

    with gr.Tab("🛠️ System Audit"):
        gr.Markdown("### 🚩 Adversarial Red-Team Audit")
        audit_btn = gr.Button("🚀 Start Security Audit")
        audit_logs = gr.Code(label="Live Audit Logs")
        audit_btn.click(fn=run_deepeval_audit, outputs=audit_logs)

# 3. Secure Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )