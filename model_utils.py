#!/usr/bin/env python
# ============================================================
#  model_utils.py  –  BERT model wrapper + prediction helpers
# ============================================================

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from config import (
    FALLBACK_HF_MODEL,
    LABEL_MAP,
    MAX_LENGTH,
    MODEL_SAVE_DIR,
    NUM_LABELS,
    PRETRAINED_MODEL,
)
from utils.preprocess import preprocess

logger = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── SentimentModel ────────────────────────────────────────────
class SentimentModel:
    """
    Wraps a HuggingFace model for 3-class sentiment inference.

    Priority:
      1. Load fine-tuned BERT from MODEL_SAVE_DIR  (if it exists)
      2. Fall back to cardiffnlp twitter-roberta   (pre-trained, no fine-tuning needed)
    """

    def __init__(self) -> None:
        self.model     : Optional[AutoModelForSequenceClassification] = None
        self.tokenizer : Optional[AutoTokenizer]                      = None
        self.model_name: str = ""
        self._load()

    # ── Loading ──────────────────────────────────────────────
    def _load(self) -> None:
        if self._fine_tuned_exists():
            self._load_fine_tuned()
        else:
            logger.info(
                "Fine-tuned model not found. Using pre-trained fallback: %s",
                FALLBACK_HF_MODEL,
            )
            self._load_fallback()

    def _fine_tuned_exists(self) -> bool:
        config_path = os.path.join(MODEL_SAVE_DIR, "config.json")
        return os.path.isfile(config_path)

    def _load_fine_tuned(self) -> None:
        logger.info("Loading fine-tuned BERT from: %s", MODEL_SAVE_DIR)
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_DIR)
        self.model     = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
        self.model.to(DEVICE).eval()
        self.model_name = "Fine-tuned BERT (bert-base-uncased)"

    def _load_fallback(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_HF_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_HF_MODEL
        )
        self.model.to(DEVICE).eval()
        self.model_name = f"Pre-trained ({FALLBACK_HF_MODEL})"

    # ── Inference ────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, text: str, apply_preprocessing: bool = True) -> dict:
        """
        Returns:
          {
            "label":       "Positive" | "Neutral" | "Negative",
            "label_id":    int,
            "confidence":  float  (0-1),
            "scores":      {"Positive": float, "Neutral": float, "Negative": float},
            "clean_text":  str,
          }
        """
        clean = preprocess(text) if apply_preprocessing else text

        enc = self.tokenizer(
            clean,
            truncation      = True,
            max_length      = MAX_LENGTH,
            padding         = "max_length",
            return_tensors  = "pt",
        ).to(DEVICE)

        logits = self.model(**enc).logits            # (1, num_labels)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()

        # ── Normalise label order ──────────────────────────
        # cardiffnlp model: 0=negative, 1=neutral, 2=positive  (same as LABEL_MAP)
        # fine-tuned BERT:  same order by construction in train.py
        label_id   = int(np.argmax(probs))
        label      = LABEL_MAP[label_id]
        confidence = float(probs[label_id])

        scores = {
            LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)
        }

        return {
            "label":      label,
            "label_id":   label_id,
            "confidence": confidence,
            "scores":     scores,
            "clean_text": clean,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        apply_preprocessing: bool = True,
        batch_size: int = 32,
    ) -> list[dict]:
        """Predict sentiment for a list of texts (batched for efficiency)."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            clean = [preprocess(t) if apply_preprocessing else t for t in chunk]

            enc = self.tokenizer(
                clean,
                truncation      = True,
                max_length      = MAX_LENGTH,
                padding         = True,
                return_tensors  = "pt",
            ).to(DEVICE)

            logits = self.model(**enc).logits
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

            for j, row_probs in enumerate(probs):
                label_id   = int(np.argmax(row_probs))
                results.append(
                    {
                        "text":       chunk[j],
                        "label":      LABEL_MAP[label_id],
                        "label_id":   label_id,
                        "confidence": round(float(row_probs[label_id]), 4),
                        "scores": {
                            LABEL_MAP[k]: round(float(p), 4)
                            for k, p in enumerate(row_probs)
                        },
                    }
                )
        return results


# ── Singleton (lazy) ─────────────────────────────────────────
_model_instance: Optional[SentimentModel] = None


def get_model() -> SentimentModel:
    """Return a cached SentimentModel instance (loads once per process)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = get_model()
    print(f"\nUsing: {m.model_name}\n")
    texts = [
        "This laptop is absolutely amazing – best purchase of my life!",
        "The product is okay I guess, nothing special.",
        "Horrible quality, broke after one day. Terrible!",
    ]
    for t in texts:
        r = m.predict(t)
        print(f"Text      : {t}")
        print(f"Label     : {r['label']}  (conf: {r['confidence']:.2%})")
        print(f"Scores    : {r['scores']}")
        print()
