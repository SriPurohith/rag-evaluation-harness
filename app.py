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
        return "⚠️ Error: OpenAI API Key not found in Space Secrets!", pd.DataFrame()

    try:
        # Generate Answer
        answer = rag_chain.invoke(question)
        
        # Retrieve Contexts for Evaluation
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]
        
        # Define Ground Truth (For demo purposes, you can use the question as a pivot 
        # or leave it as a general policy reference)
        ground_truth = "Refer to the standard corporate policy document for verification."
        
        # Run RAGAS Evaluation
        report = run_evaluation(question, answer, contexts, ground_truth)
        
        # Extract specific scores for the UI table
        metrics_data = {
            "Metric": ["Faithfulness", "Answer Relevancy", "Answer Correctness"],
            "Score": [
                round(report['faithfulness'].iloc[0], 2),
                round(report['answer_relevancy'].iloc[0], 2),
                round(report['answer_correctness'].iloc[0], 2)
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        return answer, metrics_df

    except Exception as e:
        return f"Error during processing: {str(e)}", pd.DataFrame()

# 2. Build the UI Layout (Gradio 6.0 standard)
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Policy-QA Eval Harness")
    gr.Markdown(
        "This system uses **RAGAS** to mathematically validate the accuracy of AI responses "
        "against the uploaded Corporate Policy PDF."
    )
    
    with gr.Tab("💬 Chat"):
        with gr.Column():
            input_text = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What are the internet speed requirements for remote work?"
            )
            output_text = gr.Textbox(label="AI Response", interactive=False)
            btn = gr.Button("Analyze Policy", variant="primary")

    with gr.Tab("📊 Quality Metrics"):
        gr.Markdown("### Automated Reliability Scores")
        gr.Markdown(
            "Values range from **0.0 to 1.0**. Higher is better. "
            "Faithfulness measures if the answer is derived solely from the source document."
        )
        output_table = gr.DataFrame(label="RAGAS Evaluation Results")

    # Event Mapping
    btn.click(
        fn=predict, 
        inputs=input_text, 
        outputs=[output_text, output_table]
    )

# 3. Launching with Gradio 6.0 Parameters (Fixed)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )