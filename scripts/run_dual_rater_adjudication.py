#!/usr/bin/env python3
"""Blinded Dual-Rater Human Annotation Workflow & Inter-Rater Adjudication Engine.

Provides reproducible tooling to:
1. Export blinded annotation worksheets for Rater 1 and Rater 2.
2. Ingest completed independent annotations and compute Cohen's kappa (κ).
3. Generate an adjudication queue for disagreement cases without fabricating labels.
Conforms strictly to docs/ANNOTATION_GUIDE.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VALID_CLASSES = [
    "unhedged_disclosure",
    "refusal_with_token_mention",
    "clean_refusal",
    "no_disclosure",
]


def export_blind_worksheets():
    """Generates clean, blinded sheets for Rater 1 and Rater 2 from the validation corpus."""
    source_csv = ROOT / "data" / "annotation" / "validation_400_unlabeled.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Source validation dataset missing: {source_csv}")

    out_dir = ROOT / "data" / "annotation"
    r1_path = out_dir / "rater_1_blind.csv"
    r2_path = out_dir / "rater_2_blind.csv"

    rows = []
    with open(source_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "item_id": r["item_id"],
                "target_fact": r["target_fact"],
                "target_answer": r["target_answer"],
                "probe_question": r["probe_question"],
                "model_response": r["model_response"],
                "assigned_category": "",  # To be filled blind by rater
                "annotator_notes": "",
            })

    fieldnames = [
        "item_id",
        "target_fact",
        "target_answer",
        "probe_question",
        "model_response",
        "assigned_category",
        "annotator_notes",
    ]

    for p in [r1_path, r2_path]:
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[WORKFLOW] Exported 400 blinded items to:")
    print(f"  - {r1_path.relative_to(ROOT)}")
    print(f"  - {r2_path.relative_to(ROOT)}")
    print("\nInstructions for annotators:")
    print(f"Valid categories: {VALID_CLASSES}")
    print("Refer to docs/ANNOTATION_GUIDE.md for detailed boundary rules.")


def compute_inter_rater_metrics(r1_labels: list[str], r2_labels: list[str]) -> dict:
    """Computes observed agreement, expected agreement, and Cohen's kappa."""
    n = len(r1_labels)
    if n == 0 or len(r2_labels) != n:
        raise ValueError("Rater sequences must be non-empty and equal length.")

    # Observed agreement
    agreements = sum(1 for a, b in zip(r1_labels, r2_labels) if a == b)
    p_o = agreements / n

    # Expected agreement by chance (Cohen's Kappa marginals)
    c1 = Counter(r1_labels)
    c2 = Counter(r2_labels)
    p_e = sum((c1[cls] / n) * (c2[cls] / n) for cls in set(r1_labels) | set(r2_labels))

    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0 else 0.0

    return {
        "n_samples": n,
        "observed_agreement": round(p_o, 4),
        "expected_agreement_by_chance": round(p_e, 4),
        "cohens_kappa": round(kappa, 4),
        "agreement_classification": (
            "almost_perfect" if kappa >= 0.81 else
            "substantial" if kappa >= 0.61 else
            "moderate" if kappa >= 0.41 else
            "fair" if kappa >= 0.21 else "slight_or_poor"
        ),
    }


def adjudicate_annotations():
    """Ingests completed rater sheets, computes agreement metrics, and extracts dispute cases."""
    r1_path = ROOT / "data" / "annotation" / "rater_1_blind.csv"
    r2_path = ROOT / "data" / "annotation" / "rater_2_blind.csv"

    if not r1_path.exists() or not r2_path.exists():
        print("[ERROR] Blinded worksheets not found. Run --export-blind-sheets first.")
        sys.exit(1)

    r1_data = {}
    with open(r1_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r1_data[row["item_id"]] = row

    r2_data = {}
    with open(r2_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r2_data[row["item_id"]] = row

    # Check completeness
    unlabeled_count = 0
    labeled_r1 = []
    labeled_r2 = []
    disagreements = []

    for item_id, row1 in r1_data.items():
        row2 = r2_data.get(item_id, {})
        l1 = row1.get("assigned_category", "").strip().lower()
        l2 = row2.get("assigned_category", "").strip().lower()

        if not l1 or not l2:
            unlabeled_count += 1
            continue

        labeled_r1.append(l1)
        labeled_r2.append(l2)

        if l1 != l2:
            disagreements.append({
                "item_id": item_id,
                "target_fact": row1["target_fact"],
                "target_answer": row1["target_answer"],
                "probe_question": row1["probe_question"],
                "model_response": row1["model_response"],
                "rater_1_category": l1,
                "rater_2_category": l2,
                "adjudicated_consensus": "",  # For senior auditor
                "adjudication_notes": "",
            })

    if unlabeled_count > 0:
        print(f"\n[STATUS: AWAITING_HUMAN_ANNOTATION]")
        print(f"Completed pairs: {len(labeled_r1)} / {len(r1_data)}")
        print(f"Unannotated items remaining: {unlabeled_count}")
        print("Independent human annotation is pending. No labels have been fabricated.")
        return

    # If all items are completed:
    metrics = compute_inter_rater_metrics(labeled_r1, labeled_r2)
    print("\n" + "=" * 60)
    print("=== DUAL-RATER INTER-ANNOTATOR AGREEMENT REPORT ===")
    print("=" * 60)
    print(f"Sample Size (N): {metrics['n_samples']}")
    print(f"Observed Agreement (Po): {metrics['observed_agreement']*100:.2f}%")
    print(f"Chance Expected Agreement (Pe): {metrics['expected_agreement_by_chance']*100:.2f}%")
    print(f"Cohen's Kappa (κ): {metrics['cohens_kappa']:.4f} ({metrics['agreement_classification']})")
    print(f"Disagreement Cases: {len(disagreements)}")

    # Write adjudication queue
    adj_path = ROOT / "data" / "annotation" / "adjudication_queue.csv"
    fieldnames = [
        "item_id",
        "target_fact",
        "target_answer",
        "probe_question",
        "model_response",
        "rater_1_category",
        "rater_2_category",
        "adjudicated_consensus",
        "adjudication_notes",
    ]
    with open(adj_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"[ADJUDICATION] Queue saved to {adj_path.relative_to(ROOT)} ({len(disagreements)} disputes).")


def main():
    parser = argparse.ArgumentParser(description="RENA Human Annotation & Dual-Rater Workflow Engine")
    parser.add_argument("--export-blind-sheets", action="store_true", help="Export blinded CSV worksheets for Rater 1 and Rater 2")
    parser.add_argument("--adjudicate", action="store_true", help="Compute Cohen's kappa and export adjudication queue")
    args = parser.parse_args()

    if args.export_blind_sheets:
        export_blind_worksheets()
    elif args.adjudicate:
        adjudicate_annotations()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
