import gradio as gr
import os
import pandas as pd
import subprocess
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

os.environ["DEEPEVAL_TELEMETRY"] = "False"
os.environ["PYTHONPATH"] = "."  # Helps deepeval find your 'src' folder

# Initialize RAG
try:
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain, retriever = None, None

def run_deepeval_audit():
    try:
        # Change from ["python", "-m", "deepeval", ...] to this:
        result = subprocess.run(
            ["deepeval", "test", "run", "test_deepeval.py"],
            capture_output=True,
            text=True,
            timeout=120 
        )
        return result.stdout if result.returncode == 0 else f"⚠️ Audit Error:\n{result.stderr}"
    except Exception as e:
        return f"❌ Failed to trigger audit: {str(e)}"
    
def predict(question):
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OpenAI API Key Missing in Secrets", pd.DataFrame()

    try:
        # 1. Generate & Retrieve
        raw_answer = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

        # 2. DeepEval Guardrail (Explainable Security)
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

        # 4. Security Decision Logic
        # Block if the hallucination score is TOO HIGH (greater than 0.5)
        if hallucination_metric.score > 0.5: 
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
                "✅ Pass" if hallucination_metric.score <= 0.5 else "❌ Fail",  
                "✅ Pass" if ragas_report['faithfulness'].iloc[0] >= 0.4 else "⚠️ Low",
                "N/A"
            ]
        }
        return final_answer, pd.DataFrame(metrics_data)

    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

# UI Layout Construction
with gr.Blocks(title="Policy-QA Eval Harness") as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    gr.Markdown("Enterprise RAG system featuring real-time **RAGAS** grounding and **DeepEval** security audits.")
    
    with gr.Tab("💬 Secure Chat"):
        input_box = gr.Textbox(label="Ask a Policy Question", placeholder="e.g., What is the travel reimbursement policy?")
        chat_btn = gr.Button("Submit", variant="primary")
        output_text = gr.Markdown(label="Response")
    
    with gr.Tab("📊 Quality Engineering"):
        gr.Markdown("### 📈 Real-Time Evaluation Metrics")
        gr.Markdown("This table updates automatically after every chat response.")
        metrics_table = gr.DataFrame() 

    with gr.Tab("🛠️ System Audit"):
        gr.Markdown("### 🚩 Adversarial Red-Team Audit")
        gr.Markdown("Triggers the `test_deepeval.py` suite to check for prompt injections.")
        audit_btn = gr.Button("🚀 Start Security Audit")
        audit_logs = gr.Code(label="Live Audit Logs", language="markdown", lines=15)

    # Event Wiring (Placed at the bottom to ensure all components are defined)
    chat_btn.click(
        fn=predict, 
        inputs=input_box, 
        outputs=[output_text, metrics_table]
    )
    
    audit_btn.click(
        fn=run_deepeval_audit, 
        outputs=audit_logs
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )