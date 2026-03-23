"""Compute recognition metrics: WER, CER."""

import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Levenshtein distance on words
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m] / n


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m] / n


def evaluate_recognition(gt_data: dict, predictions: dict) -> dict:
    """Evaluate recognition quality.

    Args:
        gt_data: output of parse_cvat_xml_with_text()
        predictions: {filename: "full page text"}

    Returns:
        Dict with per-image and aggregate WER/CER.
    """
    total_ref_words = 0
    total_edit_distance = 0
    per_image = {}

    for filename, gt_info in gt_data.items():
        # Build reference text from GT boxes sorted by position (top to bottom)
        sorted_boxes = sorted(gt_info["boxes"], key=lambda b: (b["ytl"], b["xtl"]))
        ref_text = " ".join(b["text"] for b in sorted_boxes if b["text"].strip())

        hyp_text = predictions.get(filename, "")
        wer = compute_wer(ref_text, hyp_text)

        ref_words = normalize_text(ref_text).split()
        hyp_words = normalize_text(hyp_text).split()
        total_ref_words += len(ref_words)

        # Count edit distance for micro-average
        n = len(ref_words)
        m = len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        total_edit_distance += dp[n][m]

        per_image[filename] = {
            "wer": round(wer, 4),
            "ref_words": len(ref_words),
            "hyp_words": len(hyp_words),
        }

    micro_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 0
    macro_wer = sum(v["wer"] for v in per_image.values()) / len(per_image) if per_image else 0

    return {
        "micro_wer": round(micro_wer, 4),
        "macro_wer": round(macro_wer, 4),
        "total_ref_words": total_ref_words,
        "per_image": per_image,
    }
