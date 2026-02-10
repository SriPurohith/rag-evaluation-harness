---
title: Policy QA Eval Harness
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
python_version: 3.11
app_file: app.py
pinned: false
---
# 🛡️ Enterprise RAG Evaluation & Safety Harness
Automated quality engineering for RAG systems using RAGAS and Pytest.


**Project Focus:** Production-Grade AI Observability, Policy Compliance, and Automated QA

## 📖 The Vision

In a production environment, "it feels accurate" is not a reliable metric. This project transforms a standard RAG (Retrieval-Augmented Generation) pipeline into a **verifiable software product**. By subjecting the AI to rigorous factual, safety, and ethical audits, I created a system that doesn't just generate text—it provides **mathematical proof of its reliability**.

---

## 🛠️ Tech Stack

* **Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB
* **LLM:** GPT-4o-mini
* **Evaluation Framework:** RAGAS (Faithfulness, Relevancy, Correctness, Precision, Recall)
* **Quality Assurance:** Pytest (Parameterization & Fixtures)

---

## 📈 Quality Dashboard (Latest Audit)

I implemented **Programmatic Quality Gates**. In this framework, if the AI's "Faithfulness" or "Correctness" drops below a defined threshold (e.g., 0.8), the deployment pipeline is flagged.

| Test Category | Metric Goal | Result | Outcome |
| --- | --- | --- | --- |
| **Factual Accuracy** | High Correctness | **0.99** | ✅ Pass |
| **Boundary Logic** | Precise Values | **1.00** | ✅ Pass |
| **Hallucination Control** | High Faithfulness | **1.00** | ✅ Pass |
| **Safety / Injection** | Blocked Refusal | **Refused** | ✅ Pass |
| **Bias / Fairness** | Uniform Application | **Neutral** | ✅ Pass |

---

## 🧠 Engineering Challenges & Solutions

### 1. The "Safe Refusal" Paradox

**Challenge:** During testing, "Security" and "Negative" test cases were returning **0.0 Relevancy** scores despite giving the correct "I don't know" response.
**Solution:** I refactored the evaluation logic to use **Conditional Thresholding**. For safety-related queries, the harness prioritizes **Answer Correctness** (intent) over **Semantic Relevancy** (embedding distance). This ensures safe models aren't unfairly penalized for correctly refusing harmful prompts.

### 2. Brittle Unit Tests & Linguistic Variance

**Challenge:** Initial automated tests failed because the AI used "eligibility" (noun) instead of "eligible" (adjective), causing strict string-match failures.
**Solution:** I migrated the QA suite to a **Keyword-Density & Intent-Matching** approach. This allows for natural language variance while maintaining a strict guardrail on the factual core of the response.

---

## 🚀 How to Run the Audit

1. **Initialize the Environment:**
```powershell
pip install -r requirements.txt

```


2. **Run the Full Test Suite:**
```powershell
python -m pytest -s tests/test_rag_pipeline.py

```



---

## 🎯 Conclusion

This project demonstrates the ability to bridge the gap between **Generative AI** and **Traditional Software Engineering**. By focusing on **Observability (RAGAS)** and **Safety (Adversarial Testing)**, I’ve built a blueprint for AI solutions that are compliant, secure, and ready for enterprise-scale deployment.

---
## **Sri Purohith**

**Date:** February 9, 2026

**LinkedIn:** [linkedin.com/in/sripurohith](https://www.google.com/search?q=https://www.linkedin.com/in/sripurohith)