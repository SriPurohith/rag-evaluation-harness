import os
import nest_asyncio
from dotenv import load_dotenv
import gradio as gr

# 1. LangChain & Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 2. Evaluation & Datasets
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from datasets import Dataset 

# 3. Ragas
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

# Fix for Gradio + Ragas async conflict
nest_asyncio.apply()

# --- INITIALIZATION ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# Load the Semantic-Aware Vector Store
vectorstore = Chroma(
    persist_directory="data/chroma_db_multi", 
    embedding_function=embeddings
)

# --- SELF-QUERY RETRIEVER ---
metadata_info = [
    AttributeInfo(name="state", description="The US state", type="string"),
    AttributeInfo(name="year", description="The policy year", type="integer"),
]

# --- ADVANCED DIVERSE RETRIEVER ---
retriever = SelfQueryRetriever.from_llm(
    llm=llm, 
    vectorstore=vectorstore, 
    document_contents="Employee policy documents", 
    metadata_field_info=metadata_info,
    search_type="mmr", 
    search_kwargs={
        "k": 5,                # Retrieve 5 total chunks for the LLM
        "fetch_k": 15,         # Fetch 15 candidates before applying MMR
        "lambda_mult": 0.25    # #FIX#: Push for aggressive diversity
    }
)

# --- ADVANCED AUDITOR PROMPT ---
# Uses a clear delimiter "### Final Answer:" to separate logic from output
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict Enterprise Policy Auditor.
    
    STEP-BY-STEP REASONING:
    1. CATEGORIZE: Identify the core topic (e.g., PTO, Internet).
    2. FILTER: Find only rules for that specific topic.
    3. VALIDATE: Check if nearby rules belong to a DIFFERENT category.
    4. INTERPRET: Use professional HR terminology. If the context mentions 'review of status' or 'termination', 
       categorize these as 'disciplinary actions' or 'penalties'.
    Show your thoughts first, then provide the final answer after the tag '### Final Answer:'.
    
    Context: {context}"""),
    ("human", "{input}"),
])

qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

def secure_policy_search(query):
    # 1. RAG Invocation
    result = qa_chain.invoke({"input": query})
    raw_response = result["answer"]
    contexts = [d.page_content for d in result["context"]]
    if not contexts:
        contexts = ["No relevant policy context found in the database."]
    # 2. Split Reasoning from Answer (Critical for Faithfulness Score)
    if "### Final Answer:" in raw_response:
        reasoning, clean_answer = raw_response.split("### Final Answer:", 1)
        reasoning = reasoning.strip()
        clean_answer = clean_answer.strip()
    else:
        reasoning = "Direct response provided."
        clean_answer = raw_response.strip()
    # 3. DeepEval Hallucination Check

    test_case = LLMTestCase(
        input=query, 
        actual_output=clean_answer, 
        context=contexts  # Use 'context' instead of 'retrieval_context'
    )
    halluc_metric = HallucinationMetric(threshold=0.5)
    halluc_metric.measure(test_case)

    # 4. RAGAS Evaluation (Evaluating only the CLEAN answer)
    data = {
        "question": [query],
        "answer": [clean_answer],
        "contexts": [contexts],
        "ground_truth": [clean_answer]
    }
    dataset = Dataset.from_dict(data)
    
    try:
        ragas_result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm)
        r_faithfulness = ragas_result['faithfulness']
        r_relevancy = ragas_result['answer_relevancy']
    except Exception as e:
        print(f"Ragas Error: {e}")
        r_faithfulness, r_relevancy = 0.0, 0.0

    # 5. Triple-Guardrail Security Logic
    is_secure = (halluc_metric.score < 0.5 and r_faithfulness > 0.7 and r_relevancy > 0.8)
    security_status = "🛡️ SECURE" if is_secure else "⚠️ AUDIT ALERT"

    return clean_answer, reasoning, security_status, halluc_metric.score, r_faithfulness, r_relevancy, contexts[0]

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 Enterprise Policy Control Center")
    
    with gr.Tabs():
        with gr.Tab("💬 Secure Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(label="Policy Question")
                    chat_output = gr.Textbox(label="AI Response", lines=4)
                    btn = gr.Button("Analyze & Audit", variant="primary")
                with gr.Column(scale=1):
                    reasoning_box = gr.Textbox(label="🔍 Auditor Reasoning", lines=8)
                    security_label = gr.Label(label="Security Audit Status")
            
        with gr.Tab("📊 Quality Engineering"):
            with gr.Row():
                halluc_score = gr.Number(label="Hallucination (DeepEval)")
                faith_score = gr.Number(label="Faithfulness (Ragas)")
                relevancy_score = gr.Number(label="Relevancy (Ragas)")
                
        with gr.Tab("🛠️ System Audit"):
            audit_log = gr.Textbox(label="Primary Source Text", lines=10)

    btn.click(
        secure_policy_search, 
        inputs=query_input, 
        outputs=[chat_output, reasoning_box, security_label, halluc_score, faith_score, relevancy_score, audit_log]
    )

if __name__ == "__main__":
    demo.launch()