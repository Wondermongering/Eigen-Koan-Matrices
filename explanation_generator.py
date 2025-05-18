# Explanation Generator - Produces human readable summaries of experiment analysis
# ---------------------------------------------------------------

from typing import Dict, List
from statistics import NormalDist


def _corr_p_value(r: float, n: int) -> float:
    """Return two-tailed p-value for correlation using t approximation."""
    if n <= 2 or r >= 1 or r <= -1:
        return float('nan')
    t = r * ((n - 2) ** 0.5) / ((1 - r ** 2) ** 0.5)
    # approximate p-value using normal distribution
    p = 2 * (1 - NormalDist().cdf(abs(t)))
    return p


def generate_explanation(analysis: Dict, alpha: float = 0.05) -> str:
    """Generate a template-based explanation from analysis data."""
    matrix = analysis.get("matrix_name", "Unknown Matrix")
    model = analysis.get("model_name", "Unknown Model")
    lines: List[str] = []
    lines.append(f"### Experiment Explanation: {model} on {matrix}")

    # Number of responses
    if "response_count" in analysis:
        lines.append(f"Number of responses analysed: {analysis['response_count']}")

    # Sentiment correlations
    corrs = analysis.get("sentiment_correlations", {})
    n = len(analysis.get("sentiment_scores", []))
    mapping = {
        "main_diag_vs_vader_pos": ("main diagonal strength", "positive sentiment"),
        "main_diag_vs_vader_neg": ("main diagonal strength", "negative sentiment"),
        "anti_diag_vs_vader_pos": ("anti-diagonal strength", "positive sentiment"),
        "anti_diag_vs_vader_neg": ("anti-diagonal strength", "negative sentiment"),
    }
    for key, (diag, senti) in mapping.items():
        if key in corrs:
            r = corrs[key]
            p = _corr_p_value(r, n)
            sig = "significant" if p < alpha else "not significant"
            lines.append(f"Correlation between {diag} and {senti}: r={r:.2f} (p={p:.3f}, {sig})")

    # Top words
    wf = analysis.get("word_frequencies")
    if wf:
        top_words = ", ".join(list(wf)[:5])
        lines.append(f"Most frequent words: {top_words}")

    # Metacommentary patterns summary
    meta = analysis.get("metacommentary_analysis", [])
    if meta:
        def _count(field: str) -> int:
            return sum(1 for m in meta if m.get(field))
        diff = _count("constraint_difficulty")
        emo = _count("emotional_detection")
        prio = _count("priority_elements")
        deprio = _count("deprioritized_elements")
        lines.append(
            f"Metacommentary mentions - difficulty: {diff}, emotion: {emo}, prioritization: {prio}, deprioritization: {deprio}"
        )

    return "\n".join(lines)
