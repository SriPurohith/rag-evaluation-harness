import gradio as gr
import pandas as pd
from src.rag_system import initialize_rag
from src.evaluator import run_evaluation

# 1. Initialize System
rag_chain, retriever = initialize_rag("data/company_policy.pdf")

def predict(question):
    # Get AI Answer
    answer = rag_chain.invoke(question)
    
    # Get RAGAS Metrics
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    # We use a placeholder for ground truth in the UI or let user provide it
    ground_truth = "The policy requires 50 Mbps internet speed." 
    
    report = run_evaluation(question, answer, contexts, ground_truth)
    
    # Format metrics for display
    metrics_df = report[['faithfulness', 'answer_relevancy', 'answer_correctness']]
    return answer, metrics_df

# 2. Build the UI with Tabs
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    gr.Markdown("Ask a question about the corporate policy and see the mathematical reliability scores.")
    
    with gr.Tab("💬 Chat"):
        input_text = gr.Textbox(label="Your Question", placeholder="e.g., What is the internet speed requirement?")
        output_text = gr.Textbox(label="AI Response")
        btn = gr.Button("Submit")

    with gr.Tab("📊 Quality Metrics"):
        gr.Markdown("### RAGAS Evaluation Results")
        output_table = gr.DataFrame(label="Metric Scores (0.0 to 1.0)")

    # Connect the button to both outputs
    btn.click(fn=predict, inputs=input_text, outputs=[output_text, output_table])

if __name__ == "__main__":
    demo.launch()