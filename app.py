import gradio as gr
import os
import pandas as pd
import subprocess
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

os.environ["DEEPEVAL_TELEMETRY"] = "False"
os.environ["PYTHONPATH"] = os.getcwd()

# Initialize RAG
try:
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain, retriever = None, None

def run_deepeval_audit():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_file_path = os.path.join(current_dir, "test_deepeval.py")
        
        result = subprocess.run(
            ["deepeval", "test", "run", test_file_path],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=current_dir,
            env={**os.environ, "TERM": "dumb"}
        )
        
        # In testing, exit code 1 just means 'some tests failed'. 
        # We still want to show the table!
        if result.stdout and "Test Results" in result.stdout:
            return result.stdout
        
        return f"⚠️ Audit System Error:\n{result.stderr}"
            
    except Exception as e:
        return f"❌ Failed to launch audit: {str(e)}"
    
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
    gr.Markdown("Enterprise RAG system with real-time **RAGAS** grounding and **DeepEval** security audits.")
    
    with gr.Tab("💬 Secure Chat"):
        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(label="Ask a Policy Question", placeholder="Type here...")
                chat_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                gr.Examples(
                    examples=[
                        ["What are the eligibility rules for Full Remote status?"], # Should Pass
                        ["Write a short poem about office coffee."], # Should be Blocked
                        ["What is the policy for working from the Moon?"] # Should be Blocked/Hallucination
                    ],
                    inputs=input_box,
                    label="Try these test cases:"
                )
        
        output_text = gr.Markdown(label="Response")
    
    with gr.Tab("📊 Quality Engineering"):
        gr.Markdown("### 📈 Real-Time Evaluation Metrics")
        gr.Markdown("This table updates automatically after every chat response to show Faithfulness and Hallucination scores.")
        metrics_table = gr.DataFrame() 

    with gr.Tab("🛠️ System Audit"):
        gr.Markdown("### 🚩 Adversarial Red-Team Audit")
        gr.Markdown("Run the full `deepeval` suite to stress-test the system against injections.")
        audit_btn = gr.Button("🚀 Start Security Audit")
        audit_logs = gr.Code(label="Live Audit Logs", language="markdown", lines=20)

    # --- Event Wiring ---
    # Force the progress animation to show while the AI and Evaluators are thinking
    chat_btn.click(
        fn=predict, 
        inputs=input_box, 
        outputs=[output_text, metrics_table],
        show_progress="full" # This ensures the loading spinner and bar are visible
    )
    
    audit_btn.click(
        fn=run_deepeval_audit, 
        outputs=audit_logs,
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )