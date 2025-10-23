import streamlit as st
import pdfplumber, re
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

st.set_page_config(page_title="AI Résumé Matcher", layout="centered")
st.title("AI-Powered Résumé Screener & Job Matcher")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = load_model()

@st.cache_data
def load_vocab():
    vocab_path = Path("data/skills_vocabulary.txt")
    if vocab_path.exists():
        return [s.strip().lower() for s in vocab_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    return ["python","pandas","numpy","sql","pytorch","tensorflow","scikit-learn","docker","airflow",
            "aws","gcp","azure","fastapi","streamlit","nlp","computer vision","time series",
            "spark","dbt","power bi","tableau","mlops","kubernetes","git","linux"]

SKILLS = load_vocab()

def read_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        return "
".join([p.extract_text() or "" for p in pdf.pages])

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()

def cosine_sim(a, b) -> float:
    return float(util.cos_sim(a, b).item())

def extract_skills(text: str, skills_vocab):
    found=[]
    t = " " + clean(text) + " "
    for s in skills_vocab:
        if f" {s} " in t:
            found.append(s)
    return sorted(set(found))

def estimate_years(t: str) -> int:
    nums = re.findall(r"(\d+)\+?\s*years", t)
    return max([int(x) for x in nums], default=0)

st.markdown("Upload your **résumé (PDF)** and paste a **job description** to get a match score with explanations.")

cv_file = st.file_uploader("Upload résumé (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description (text)", height=220, placeholder="Paste the JD here...")

colA, colB = st.columns(2)
with colA:
    show_debug = st.checkbox("Show debug metrics", value=False)
with colB:
    run_btn = st.button("Run Match")

if run_btn and cv_file and jd_text.strip():
    try:
        cv_text = clean(read_pdf(cv_file))
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
        st.stop()

    jd = clean(jd_text)

    emb_cv = model.encode(cv_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_jd = model.encode(jd, convert_to_tensor=True, normalize_embeddings=True)
    sim_text = cosine_sim(emb_cv, emb_jd)

    cv_sk = extract_skills(cv_text, SKILLS)
    jd_sk = extract_skills(jd, SKILLS)

    matched = sorted(list(set(cv_sk) & set(jd_sk)))
    missing = sorted(list(set(jd_sk) - set(cv_sk)))
    union = set(cv_sk) | set(jd_sk)

    skills_cov = (len(matched) / max(1, len(set(jd_sk))))  # 0..1
    tool_overlap = (len(matched) / max(1, len(union)))     # 0..1

    cv_years = estimate_years(cv_text)
    jd_years = estimate_years(jd)
    years_delta = abs(cv_years - jd_years)
    years_term = 1 - min(years_delta/5, 1)  # penalize if very off

    score = 0.60*sim_text + 0.20*skills_cov + 0.10*tool_overlap + 0.10*years_term
    score = int(round(100*max(0.0, min(1.0, score))))

    st.subheader(f"Match Score: {score}/100")
    st.write("**Matched skills:**", matched if matched else "—")
    st.write("**Missing skills to address:**", missing if missing else "—")

    if show_debug:
        st.divider()
        st.markdown("**Debug Metrics**")
        st.write(f"Sim(text): `{sim_text:.2f}`  |  SkillsCoverage: `{skills_cov:.2f}`  |  ToolOverlap: `{tool_overlap:.2f}`  |  YearsTerm: `{years_term:.2f}`")
        st.write(f"CV years (heuristic): `{cv_years}`, JD years (heuristic): `{jd_years}`")
        st.write(f"CV skills found ({len(cv_sk)}): {cv_sk}")
        st.write(f"JD skills found ({len(jd_sk)}): {jd_sk}")
elif run_btn:
    st.warning("Please upload a résumé (PDF) and paste a job description.")
