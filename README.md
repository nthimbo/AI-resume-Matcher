# AI – Resume Screening and Job Matcher

The **AI – Resume Screening and Job Matcher** is an intelligent system that automates résumé evaluation using **Natural Language Processing (NLP)** and **machine learning**. Built with **FastAPI**, **scikit-learn**, and **Sentence-Transformer embeddings**, it analyzes candidate résumés and job descriptions to produce an explainable **match score (0–100)** showing how well a candidate fits a role.

The system extracts skills, experience, and education from uploaded PDF résumés, processes job descriptions, and compares both texts through **TF-IDF vectorization**, **cosine similarity**, and **semantic embeddings**. It highlights overlapping and missing skills, identifies experience gaps, and explains each score component, enabling transparent and data-driven hiring.

## ⚙️ Core Modules
1. **Resume Parser** – Reads PDF résumés, cleans text, and extracts candidate information with NLP.  
2. **Job Description Analyzer** – Detects required skills, keywords, and experience levels.  
3. **Feature & Embedding Generator** – Converts text to numeric vectors using TF-IDF and sentence embeddings.  
4. **Scoring Engine** – Computes similarity metrics and overall job-fit score with explainable sub-scores.  
5. **Explainability Module** – Displays matched and missing skills with recommendations.  
6. **FastAPI/Streamlit Layer** – Serves endpoints/UI for web or HR-system integration.

**Stack:** Python · FastAPI · Streamlit · scikit-learn · Sentence-Transformers · TF‑IDF · Cosine Similarity · Docker

### Run locally (Streamlit MVP)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_mvp.py
```

### Deploy (Render.com / Docker)
- Build with the provided Dockerfile (exposes port 8501 for Streamlit).
- Or deploy to Streamlit Cloud by pointing to `app_mvp.py`.

### Project Objective
Apply AI/NLP to automate résumé screening with **transparency**, **fairness**, and **production-ready** design.
