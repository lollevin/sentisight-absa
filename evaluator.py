"""
Evaluation engine: compares ABSA model output against ground-truth labeled data.
Calculates aspect detection rate, sentiment accuracy, precision, recall, F1.
"""
from __future__ import annotations
import os
import pandas as pd
from collections import defaultdict


def load_labeled_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "labeled_test_data.csv")
    return pd.read_csv(path)


def group_by_review(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group labeled rows by review text."""
    groups: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        r = row["review"].strip().strip('"')
        groups.setdefault(r, []).append({
            "aspect": row["aspect"].strip().lower(),
            "sentiment": row["expected_sentiment"].strip().lower(),
        })
    return groups


def normalize_aspect(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def evaluate_results(labeled_df: pd.DataFrame, absa_results: list[dict]) -> dict:
    """
    Compare ABSA model predictions to ground-truth labels.

    Returns a dict with:
      - aspect_detection_rate: how often the model found the correct aspect
      - sentiment_accuracy: among detected aspects, how often sentiment matched
      - per_sentiment_metrics: precision/recall/F1 per sentiment class
      - overall_f1: macro F1
      - detail_rows: per-aspect comparison rows for display
    """
    groups = group_by_review(labeled_df)

    true_labels: list[str] = []
    pred_labels: list[str] = []
    detected_flags: list[bool] = []
    detail_rows: list[dict] = []

    for result in absa_results:
        review_text = result.get("review", "").strip().strip('"')
        ground_truths = groups.get(review_text, [])
        predicted_aspects = {
            normalize_aspect(a["name"]): a.get("sentiment", "neutral")
            for a in result.get("aspects", [])
        }

        for gt in ground_truths:
            gt_aspect = normalize_aspect(gt["aspect"])
            gt_sentiment = gt["sentiment"]

            # Find best matching predicted aspect (exact or substring match)
            matched_sentiment = None
            for pred_asp, pred_sent in predicted_aspects.items():
                if gt_aspect in pred_asp or pred_asp in gt_aspect:
                    matched_sentiment = pred_sent
                    break

            detected = matched_sentiment is not None
            detected_flags.append(detected)

            true_labels.append(gt_sentiment)
            pred_labels.append(matched_sentiment if detected else "not_detected")

            detail_rows.append({
                "Review": review_text[:60] + "..." if len(review_text) > 60 else review_text,
                "Aspect": gt_aspect,
                "Expected": gt_sentiment,
                "Predicted": matched_sentiment if detected else "❌ not detected",
                "Match": "✅" if detected and matched_sentiment == gt_sentiment else ("⚠️" if detected else "❌"),
            })

    total = len(true_labels)
    detected_count = sum(detected_flags)
    aspect_detection_rate = detected_count / total if total else 0

    correct_sentiment = sum(
        1 for t, p, d in zip(true_labels, pred_labels, detected_flags)
        if d and t == p
    )
    sentiment_accuracy = correct_sentiment / detected_count if detected_count else 0

    # Per-class precision / recall / F1 (ignoring "not_detected" as predicted class)
    classes = ["positive", "negative", "neutral"]
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for t in true_labels if t == cls)}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

    return {
        "total_aspects": total,
        "detected_count": detected_count,
        "aspect_detection_rate": aspect_detection_rate,
        "sentiment_accuracy": sentiment_accuracy,
        "per_sentiment_metrics": per_class,
        "overall_f1": macro_f1,
        "detail_rows": detail_rows,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }


def vader_baseline_metrics(labeled_df: pd.DataFrame, vader_results: list[dict]) -> dict:
    """
    Evaluate VADER against the same labeled data.
    VADER only predicts overall sentiment — we compare it to each aspect's ground truth
    to show its limitation.
    """
    groups = group_by_review(labeled_df)
    reviews_unique = list(groups.keys())

    true_labels: list[str] = []
    pred_labels: list[str] = []

    for vader_r, review_text in zip(vader_results, reviews_unique):
        ground_truths = groups.get(review_text, [])
        vader_label = vader_r.get("label", "neutral")
        for gt in ground_truths:
            true_labels.append(gt["sentiment"])
            pred_labels.append(vader_label)

    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / total if total else 0

    classes = ["positive", "negative", "neutral"]
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

    return {
        "accuracy": accuracy,
        "overall_f1": macro_f1,
        "per_sentiment_metrics": per_class,
        "total": total,
    }
