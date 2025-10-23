from sentence_transformers import SentenceTransformer
from match.scorer import hybrid_score
def test_score_bounds():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    a = model.encode("python pandas numpy", convert_to_tensor=True, normalize_embeddings=True)
    b = model.encode("looking for python and pandas", convert_to_tensor=True, normalize_embeddings=True)
    score, parts = hybrid_score(a, b, matched=["python","pandas"], jd_skills=["python","pandas","sql"],
                                union=set(["python","pandas","sql"]), cv_years=3, jd_years=2)
    assert 0 <= score <= 100
