"""
train.py
========
Training script for all models:
  1. Logistic Regression  (traditional ML)
  2. SVM                  (traditional ML)
  3. Random Forest        (traditional ML)
  4. LSTM                 (deep learning)
  5. DistilBERT           (pretrained transformer)

Saves all models to the models/ directory.
Logs training metrics and generates comparison plots.

Author: NLP Engineering Team
"""

import os
import json
import time
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from preprocessing import (
    TextPreprocessor, TFIDFExtractor, StorageLabelEncoder,
    DataLoader, build_vocab, texts_to_sequences,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR  = "models"
PLOTS_DIR   = "models/plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

LABEL_NAMES = ["freezer", "fridge", "normal"]

# ─── Helper: Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(       y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    logger.info(f"\n{'='*60}\n{model_name} — Accuracy={acc:.4f} F1={f1:.4f}\n{report}")
    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─── Helper: Confusion Matrix Plot ────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRADITIONAL ML MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TraditionalMLTrainer:
    """Trains Logistic Regression, SVM, and Random Forest on TF-IDF features."""

    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        self.preprocessor = TextPreprocessor()
        self.tfidf        = TFIDFExtractor(max_features=15_000, ngram_range=(1, 2))
        self.label_enc    = StorageLabelEncoder()

        self.X_train = self.tfidf.fit_transform(train_df["cleaned"].tolist())
        self.X_val   = self.tfidf.transform(val_df["cleaned"].tolist())
        self.X_test  = self.tfidf.transform(test_df["cleaned"].tolist())

        self.y_train = train_df["label_id"].values
        self.y_val   = val_df["label_id"].values
        self.y_test  = test_df["label_id"].values

        # Save TF-IDF vectorizer
        with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf.vectorizer, f)
        logger.info("TF-IDF vectorizer saved.")

    def _train_model(self, clf, name: str) -> dict:
        logger.info(f"Training {name}...")
        t0 = time.time()
        clf.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        val_preds  = clf.predict(self.X_val)
        test_preds = clf.predict(self.X_test)

        val_metrics  = compute_metrics(self.y_val,  val_preds,  f"{name} (val)")
        test_metrics = compute_metrics(self.y_test, test_preds, f"{name} (test)")

        plot_confusion_matrix(self.y_test, test_preds, name)

        # Save model
        path = f"{MODELS_DIR}/{name.replace(' ', '_').lower()}.pkl"
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        logger.info(f"{name} saved → {path} (training time: {elapsed:.1f}s)")

        return {**test_metrics, "train_time_s": round(elapsed, 2)}

    def train_all(self) -> list:
        results = []

        results.append(self._train_model(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                            random_state=SEED),
            "Logistic Regression",
        ))

        results.append(self._train_model(
            LinearSVC(max_iter=2000, C=1.0, random_state=SEED),
            "SVM",
        ))

        results.append(self._train_model(
            RandomForestClassifier(n_estimators=300, max_depth=None,
                                   random_state=SEED, n_jobs=-1),
            "Random Forest",
        ))

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  DISTILBERT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BERTDataset(Dataset):
    def __init__(self, sentences: list, labels: np.ndarray, tokenizer, max_len: int = 128):
        self.sentences = sentences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DistilBERTTrainer:
    """Fine-tunes DistilBERT for storage type classification."""

    MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN    = 128
    EPOCHS     = 5
    BATCH_SIZE = 32
    LR         = 2e-5

    def __init__(self, train_df, val_df, test_df):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.tokenizer.save_pretrained(f"{MODELS_DIR}/distilbert_tokenizer")

        self.train_loader = TorchDataLoader(
            BERTDataset(train_df["sentence"].tolist(), train_df["label_id"].values,
                        self.tokenizer, self.MAX_LEN),
            batch_size=self.BATCH_SIZE, shuffle=True,
        )
        self.val_loader = TorchDataLoader(
            BERTDataset(val_df["sentence"].tolist(), val_df["label_id"].values,
                        self.tokenizer, self.MAX_LEN),
            batch_size=self.BATCH_SIZE,
        )
        self.test_loader = TorchDataLoader(
            BERTDataset(test_df["sentence"].tolist(), test_df["label_id"].values,
                        self.tokenizer, self.MAX_LEN),
            batch_size=self.BATCH_SIZE,
        )

    def train(self) -> dict:
        model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME, num_labels=3
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=self.LR, weight_decay=0.01)
        total_steps = len(self.train_loader) * self.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        logger.info(f"Fine-tuning DistilBERT on {DEVICE}...")
        t0 = time.time()

        for epoch in range(self.EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss    = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_preds, val_true = [], []
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attn_mask = batch["attention_mask"].to(DEVICE)
                    labels    = batch["labels"].to(DEVICE)
                    outputs   = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    preds     = outputs.logits.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_true, val_preds)
            avg_val_loss = val_loss / len(self.val_loader)

            history["train_loss"].append(total_loss / len(self.train_loader))
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            logger.info(f"Epoch {epoch+1}/{self.EPOCHS} | val_acc={val_acc:.4f} | val_loss={avg_val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(f"{MODELS_DIR}/distilbert_best")

        elapsed = time.time() - t0

        # Final test evaluation
        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"]
                outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
                preds     = outputs.logits.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.numpy())

        metrics = compute_metrics(test_true, test_preds, "DistilBERT (test)")
        plot_confusion_matrix(test_true, test_preds, "DistilBERT")
        self._plot_history(history)

        return {**metrics, "train_time_s": round(elapsed, 2)}

    def _plot_history(self, history: dict):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"],   label="Val Loss")
        axes[0].set_title("DistilBERT — Loss")
        axes[0].legend()
        axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
        axes[1].set_title("DistilBERT — Validation Accuracy")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/distilbert_training_history.png", dpi=150)
        plt.close()
        logger.info("DistilBERT training history plot saved.")


# ─── Performance Comparison Plot ──────────────────────────────────────────────

def plot_comparison(results: list):
    df = pd.DataFrame(results)
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df["model"], df[metric], color=colors[:len(df)])
        ax.set_title(metric.capitalize(), fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Performance Comparison (Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved → {path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    loader = DataLoader()
    train_df = loader.load("data/train.csv")
    val_df   = loader.load("data/validation.csv")
    test_df  = loader.load("data/test.csv")

    all_results = []

    # Traditional ML
    ml_trainer = TraditionalMLTrainer(train_df, val_df, test_df)
    all_results.extend(ml_trainer.train_all())

    # DistilBERT
    bert_trainer = DistilBERTTrainer(train_df, val_df, test_df)
    all_results.append(bert_trainer.train())

    # Save comparison table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{MODELS_DIR}/results_comparison.csv", index=False)
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    plot_comparison(all_results)

    # Save results as JSON for API use
    with open(f"{MODELS_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("All models trained and saved successfully.")


if __name__ == "__main__":
    main()