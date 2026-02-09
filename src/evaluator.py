from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import pandas as pd

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Updated imports to satisfy the Deprecation Warnings
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness

from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall,
    answer_correctness
)

from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall,
    answer_correctness
)
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def run_evaluation(question, answer, contexts, ground_truth):
    # RAGAS expects a dictionary of lists
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts], # This must be a list of strings
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(data)
    
    # We use gpt-4o-mini to keep your costs very low during evaluation
    eval_llm = ChatOpenAI(model="gpt-4o-mini")
    eval_embeddings = OpenAIEmbeddings()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness, 
            answer_relevancy, 
            context_precision, 
            context_recall,
            answer_correctness
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    return result.to_pandas()