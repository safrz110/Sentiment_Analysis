#!/usr/bin/env python
# ============================================================
#  train.py  –  Fine-tune BERT for 3-class sentiment analysis
# ============================================================
#
#  Usage:
#    python train.py                        # uses data/reviews.csv
#    python train.py --data path/to/my.csv  # custom dataset
#    python train.py --epochs 6 --lr 3e-5   # override hyper-params
#
#  After training the fine-tuned model is saved to:
#    models/fine_tuned_bert/
#  and will be auto-loaded by model_utils.py on next run.
# ============================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from config import (
    BATCH_SIZE,
    DATA_FILE,
    LABEL_COLUMN,
    LABEL_MAP,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_SAVE_DIR,
    NUM_EPOCHS,
    NUM_LABELS,
    PRETRAINED_MODEL,
    RANDOM_SEED,
    TEXT_COLUMN,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from data.generate_data import save_dataset
from utils.preprocess import batch_preprocess

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "training.log")),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Dataset ──────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation     = True,
            padding        = "max_length",
            max_length     = max_len,
            return_tensors = "pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids",
                torch.zeros(self.encodings["input_ids"].shape[1], dtype=torch.long),
            )[idx],
            "labels": self.labels[idx],
        }


# ── Data loading ─────────────────────────────────────────────
def load_data(data_path: str) -> tuple[list[str], list[int]]:
    if not os.path.isfile(data_path):
        logger.info("Dataset not found. Generating synthetic data …")
        save_dataset(data_path)

    df = pd.read_csv(data_path)
    assert TEXT_COLUMN  in df.columns, f"Missing column: '{TEXT_COLUMN}'"
    assert LABEL_COLUMN in df.columns, f"Missing column: '{LABEL_COLUMN}'"

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    texts  = batch_preprocess(df[TEXT_COLUMN].tolist())
    labels = df[LABEL_COLUMN].tolist()

    logger.info("Loaded %d samples", len(texts))
    for lid, lname in LABEL_MAP.items():
        count = labels.count(lid)
        logger.info("  %-10s: %d", lname, count)

    return texts, labels


# ── Train / eval helpers ──────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler) -> tuple[float, float]:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels         = labels,
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels         = labels,
        )

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


# ── Main ─────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    logger.info("Device: %s", DEVICE)
    logger.info("Loading and preprocessing data …")

    texts, labels = load_data(args.data)

    # Train / val / test split
    test_ratio = 1 - TRAIN_SPLIT - VAL_SPLIT
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels,
        test_size    = test_ratio,
        random_state = RANDOM_SEED,
        stratify     = labels,
    )
    val_ratio_adjusted = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size    = val_ratio_adjusted,
        random_state = RANDOM_SEED,
        stratify     = y_train_val,
    )

    logger.info("Split → train: %d | val: %d | test: %d",
                len(X_train), len(X_val), len(X_test))

    # Tokenizer + datasets
    logger.info("Initialising tokenizer: %s", PRETRAINED_MODEL)
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)

    train_ds = ReviewDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds   = ReviewDataset(X_val,   y_val,   tokenizer, args.max_len)
    test_ds  = ReviewDataset(X_test,  y_test,  tokenizer, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    logger.info("Loading BERT base model …")
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels       = NUM_LABELS,
        id2label         = LABEL_MAP,
        label2id         = {v: k for k, v in LABEL_MAP.items()},
    ).to(DEVICE)

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = WEIGHT_DECAY,
    )
    total_steps   = len(train_dl) * args.epochs
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # Training loop
    best_val_acc, best_epoch = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    logger.info("=" * 55)
    logger.info("  Starting fine-tuning for %d epoch(s)", args.epochs)
    logger.info("=" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_dl, optimizer, scheduler)
        val_loss,   val_acc, _, _ = evaluate(model, val_dl)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d | train_loss %.4f | train_acc %.4f | "
            "val_loss %.4f | val_acc %.4f | %.0fs",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss,   val_acc,
            elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            logger.info("  ✅  New best model saved (val_acc=%.4f)", best_val_acc)

    # Final evaluation on test set
    logger.info("\nLoading best model (epoch %d) for test evaluation …", best_epoch)
    best_model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(DEVICE)
    _, test_acc, y_true, y_pred = evaluate(best_model, test_dl)

    logger.info("\n%s", "=" * 55)
    logger.info("  TEST RESULTS")
    logger.info("  Accuracy : %.4f  (%.2f%%)", test_acc, test_acc * 100)
    logger.info(
        "  F1 (macro): %.4f",
        f1_score(y_true, y_pred, average="macro"),
    )
    logger.info("\nClassification Report:\n%s",
                classification_report(y_true, y_pred,
                                      target_names=list(LABEL_MAP.values())))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))
    logger.info("=" * 55)
    logger.info("Model saved to: %s", MODEL_SAVE_DIR)


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    parser.add_argument("--data",       default=DATA_FILE,   help="Path to CSV dataset")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS,   help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE,help="Learning rate")
    parser.add_argument("--max_len",    type=int,   default=MAX_LENGTH,   help="Max token length")
    args = parser.parse_args()
    main(args)
