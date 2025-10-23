def explain(matched, missing):
    if missing:
        gaps = ", ".join(missing[:10])
        gap_msg = f"Consider adding evidence for: {gaps}."
    else:
        gap_msg = "No obvious missing skills from the vocabulary list."
    match_msg = f"Matched skills: {', '.join(matched[:15])}" if matched else "No matched skills found."
    return match_msg, gap_msg
