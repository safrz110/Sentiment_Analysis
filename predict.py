#!/usr/bin/env python
# ============================================================
#  predict.py  –  CLI for sentiment prediction
# ============================================================
#
#  Usage:
#    # Single text
#    python predict.py --text "This product is amazing!"
#
#    # Batch from a CSV
#    python predict.py --csv reviews_to_score.csv --out results.csv
#
#    # Interactive mode
#    python predict.py
# ============================================================

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import LABEL_COLORS, LABEL_EMOJIS
from model_utils import get_model

logging.basicConfig(level=logging.WARNING)


def _fmt_result(result: dict) -> str:
    emoji = LABEL_EMOJIS.get(result["label"], "")
    bars  = {k: "█" * int(v * 30) for k, v in result["scores"].items()}
    lines = [
        f"  Sentiment  : {emoji}  {result['label']}",
        f"  Confidence : {result['confidence']:.2%}",
        f"  Scores:",
    ]
    for lbl, bar in bars.items():
        score = result["scores"][lbl]
        lines.append(f"    {lbl:<10} {bar:<30} {score:.2%}")
    return "\n".join(lines)


def predict_single(text: str) -> None:
    model  = get_model()
    result = model.predict(text)
    print(f"\n{'─'*50}")
    print(f"  Text: {text[:80]}{'…' if len(text) > 80 else ''}")
    print(f"{'─'*50}")
    print(_fmt_result(result))
    print(f"{'─'*50}\n")


def predict_csv(csv_path: str, out_path: str, text_col: str) -> None:
    df    = pd.read_csv(csv_path)
    assert text_col in df.columns, f"Column '{text_col}' not found in CSV."

    model   = get_model()
    results = model.predict_batch(df[text_col].astype(str).tolist())

    df["sentiment"]  = [r["label"]      for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["score_pos"]  = [r["scores"].get("Positive", 0) for r in results]
    df["score_neu"]  = [r["scores"].get("Neutral",  0) for r in results]
    df["score_neg"]  = [r["scores"].get("Negative", 0) for r in results]

    df.to_csv(out_path, index=False)
    print(f"\n✅  Results saved → {out_path}  ({len(df)} rows)")
    print(df["sentiment"].value_counts().to_string())


def interactive_mode() -> None:
    model = get_model()
    print(f"\n🎯  Sentiment Analysis Engine  [{model.model_name}]")
    print("    Type text and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            continue
        result = model.predict(text)
        print(_fmt_result(result))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment analysis prediction CLI")
    parser.add_argument("--text",     type=str, help="Single text to classify")
    parser.add_argument("--csv",      type=str, help="Path to CSV file for batch prediction")
    parser.add_argument("--out",      type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--col",      type=str, default="text",            help="Text column name in CSV")
    args = parser.parse_args()

    if args.text:
        predict_single(args.text)
    elif args.csv:
        predict_csv(args.csv, args.out, args.col)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
