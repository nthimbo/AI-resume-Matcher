import re, pdfplumber

def pdf_to_text(fp) -> str:
    with pdfplumber.open(fp) as pdf:
        return "\n".join([p.extract_text() or "" for p in pdf.pages])

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()
