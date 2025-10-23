import re
from sentence_transformers import util

def estimate_years(t: str) -> int:
    nums = re.findall(r"(\d+)\+?\s*years", t)
    return max([int(x) for x in nums], default=0)

def cosine_sim(a, b) -> float:
    return float(util.cos_sim(a, b).item())

def hybrid_score(emb_cv, emb_jd, matched, jd_skills, union, cv_years, jd_years) -> (int, tuple):
    sim_text = cosine_sim(emb_cv, emb_jd)
    skills_cov = (len(matched) / max(1, len(set(jd_skills))))
    tool_overlap = (len(matched) / max(1, len(union)))
    years_term = 1 - min(abs(cv_years - jd_years)/5, 1)
    score = 0.60*sim_text + 0.20*skills_cov + 0.10*tool_overlap + 0.10*years_term
    return int(round(100*max(0.0, min(1.0, score)))), (sim_text, skills_cov, tool_overlap, years_term)
