# ============================================================
#  config.py  –  Central configuration for Sentiment Engine
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "models", "fine_tuned_bert")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR,       exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────
DATA_FILE       = os.path.join(DATA_DIR, "reviews.csv")
TEXT_COLUMN     = "text"
LABEL_COLUMN    = "label"           # 0 = negative, 1 = neutral, 2 = positive

# ── Model ────────────────────────────────────────────────────
PRETRAINED_MODEL      = "bert-base-uncased"
FALLBACK_HF_MODEL     = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NUM_LABELS            = 3
MAX_LENGTH            = 128          # max token length for BERT

# ── Training ─────────────────────────────────────────────────
TRAIN_SPLIT     = 0.80
VAL_SPLIT       = 0.10              # remaining 10 % → test
RANDOM_SEED     = 42
BATCH_SIZE      = 16
NUM_EPOCHS      = 4
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.10
FP16            = True              # mixed-precision (GPU only)

# ── Labels ───────────────────────────────────────────────────
LABEL_MAP       = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS    = {
    "Positive": "#22c55e",
    "Neutral":  "#f59e0b",
    "Negative": "#ef4444",
}
LABEL_EMOJIS    = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😞",
}
