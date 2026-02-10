import gradio as gr
import os
import pandas as pd
import re
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation

# 1. Initialize RAG System with Error Handling
try:
    # Ensure this matches your file path in the 'data' folder
    rag_chain, retriever = initialize_rag("data/company_policy.pdf")
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    rag_chain, retriever = None, None

def predict(question):
    # --- LAYER 1: Environment Check ---
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ [SYSTEM ERROR]: OpenAI API Key missing in Space Secrets.", pd.DataFrame()
    
    if rag_chain is None:
        return "⚠️ [SYSTEM ERROR]: RAG Pipeline failed to load. Check 'data' folder.", pd.DataFrame()

    try:
        # --- LAYER 2: Generate & Retrieve ---
        raw_answer = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]
        ground_truth = "Refer to official Corporate Policy document."

        # --- LAYER 3: RAGAS Evaluation ---
        report = run_evaluation(question, raw_answer, contexts, ground_truth)
        faithfulness = report['faithfulness'].iloc[0]
        relevancy = report['answer_relevancy'].iloc[0]

        # --- LAYER 4: SECURITY GUARDRAILS ---
        
        # A. Formatting Guardrail: Block JSON/Code injection attempts
        # This catches the 'System Auditor' style attacks you found
        if "{" in raw_answer or "page_" in raw_answer or "```json" in raw_answer.lower():
            final_answer = (
                "🛡️ [SECURITY BLOCK]: Unauthorized output format detected. "
                "The system is restricted to plain-text policy assistance only."
            )
        
        # B. Faithfulness Guardrail: Block Hallucinations or Instruction Overrides
        # If the AI ignores the PDF (like the poem), faithfulness drops to near zero
        elif faithfulness < 0.4:
            final_answer = (
                "🛡️ [SECURITY BLOCK]: Response failed Faithfulness check. "
                "The AI attempted to provide information not found in the official policy."
            )
            
        else:
            final_answer = raw_answer

        # --- LAYER 5: Metrics Preparation ---
        metrics_data = {
            "Metric": ["Faithfulness (Grounding)", "Answer Relevancy", "Answer Correctness"],
            "Score": [
                round(faithfulness, 2),
                round(relevancy, 2),
                round(report['answer_correctness'].iloc[0], 2)
            ],
            "Status": [
                "✅ Pass" if faithfulness >= 0.4 else "❌ Fail",
                "✅ Pass" if relevancy >= 0.7 else "⚠️ Low",
                "N/A"
            ]
        }
        return final_answer, pd.DataFrame(metrics_data)

    except Exception as e:
        return f"❌ [RUNTIME ERROR]: {str(e)}", pd.DataFrame()

# 2. UI Layout (Gradio 6.5.1 Optimized)

with gr.Blocks(title="Policy-QA Eval Harness") as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    gr.Markdown(
        "**Secure Enterprise RAG Demo.** This system uses automated RAGAS metrics to "
        "detect hallucinations and block prompt injection attacks in real-time."
    )
    
    with gr.Tab("💬 Secure Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                input_text = gr.Textbox(
                    label="Policy Query", 
                    placeholder="e.g., What is the AI usage policy?",
                    lines=2
                )
                btn = gr.Button("Analyze Request", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Safety Status")
                gr.Markdown("Active Guardrails: \n- PII Filter\n- Format Lock\n- RAGAS Verifier")

        output_text = gr.Textbox(label="Verified AI Response", interactive=False, lines=10)

    with gr.Tab("📊 Quality Engineering"):
        gr.Markdown("### Automated Evaluation Report (RAGAS)")
        output_table = gr.DataFrame()
        gr.Markdown(
            "**Note:** Responses with a Faithfulness score < 0.4 are automatically "
            "censored to prevent misinformation."
        )

    # Wire up the logic
    btn.click(
        fn=predict, 
        inputs=input_text, 
        outputs=[output_text, output_table]
    )
    gr.Examples(
        examples=[
            ["What is the residency rule for Full Remote status?"],
            ["What is the policy on pasting API keys into AI tools?"],
            ["Can I work remotely if I live in Spokane?"]
        ],
        inputs=input_text
    )
# 3. Secure Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )