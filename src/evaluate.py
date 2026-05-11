"""
evaluate.py
===========
Evaluation script for trained models. Loads saved models and computes
detailed metrics on the test set.

Author: NLP Engineering Team
"""

import pickle
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataLoader, TextPreprocessor, TFIDFExtractor, StorageLabelEncoder
from src.train import LSTMClassifier, texts_to_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["freezer", "fridge", "normal"]


class ModelEvaluator:
    """Loads and evaluates all trained models on the test set."""
    
    def __init__(self, test_df: pd.DataFrame):
        """
        Initialize evaluator with test data.
        
        Args:
            test_df: DataFrame with 'sentence' and 'storage_label' columns
        """
        self.test_df = test_df
        self.label_enc = StorageLabelEncoder()
        
    def evaluate_logistic_regression(self) -> dict:
        """Load and evaluate Logistic Regression model."""
        try:
            with open(f"{MODELS_DIR}/logistic_regression.pkl", "rb") as f:
                model = pickle.load(f)
            
            # Process test data
            preprocessor = TextPreprocessor()
            tfidf = TFIDFExtractor()
            
            # Need to load fitted vectorizer
            with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
                tfidf.vectorizer = pickle.load(f)
            
            cleaned_texts = preprocessor.transform(self.test_df["sentence"].tolist())
            X_test = tfidf.transform(cleaned_texts)
            y_true = self.label_enc.encode(self.test_df["storage_label"])
            y_pred = model.predict(X_test)
            
            return self._compute_metrics(y_true, y_pred, "Logistic Regression")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            return None
    
    def evaluate_svm(self) -> dict:
        """Load and evaluate SVM model."""
        try:
            with open(f"{MODELS_DIR}/svm.pkl", "rb") as f:
                model = pickle.load(f)
            
            preprocessor = TextPreprocessor()
            with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
            
            cleaned_texts = preprocessor.transform(self.test_df["sentence"].tolist())
            X_test = tfidf.transform(cleaned_texts)
            y_true = self.label_enc.encode(self.test_df["storage_label"])
            y_pred = model.predict(X_test)
            
            return self._compute_metrics(y_true, y_pred, "SVM")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            return None
    
    def evaluate_random_forest(self) -> dict:
        """Load and evaluate Random Forest model."""
        try:
            with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
                model = pickle.load(f)
            
            preprocessor = TextPreprocessor()
            with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
            
            cleaned_texts = preprocessor.transform(self.test_df["sentence"].tolist())
            X_test = tfidf.transform(cleaned_texts)
            y_true = self.label_enc.encode(self.test_df["storage_label"])
            y_pred = model.predict(X_test)
            
            return self._compute_metrics(y_true, y_pred, "Random Forest")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            return None
    
    
    def evaluate_distilbert(self) -> dict:
        """Load and evaluate DistilBERT model."""
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            
            tokenizer = DistilBertTokenizerFast.from_pretrained(f"{MODELS_DIR}/distilbert_tokenizer")
            model = DistilBertForSequenceClassification.from_pretrained(f"{MODELS_DIR}/distilbert_best").to(DEVICE)
            model.eval()
            
            y_true = self.label_enc.encode(self.test_df["storage_label"])
            y_pred = []
            MAX_LEN = 128
            BATCH_SIZE = 32
            
            # Process in batches
            for i in range(0, len(self.test_df), BATCH_SIZE):
                batch_sentences = self.test_df["sentence"].tolist()[i:i+BATCH_SIZE]
                encodings = tokenizer(
                    batch_sentences,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
                
                with torch.no_grad():
                    outputs = model(**encodings)
                    batch_preds = outputs.logits.argmax(dim=1).cpu().numpy()
                    y_pred.extend(batch_preds)
            
            return self._compute_metrics(y_true, y_pred, "DistilBERT")
        except Exception as e:
            logger.error(f"DistilBERT evaluation failed: {e}")
            return None
    
    def _compute_metrics(self, y_true, y_pred, model_name: str) -> dict:
        """Helper to compute and log metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_name} Evaluation")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}")
        
        return {
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    
    def evaluate_all(self) -> pd.DataFrame:
        """Run evaluation for all available models."""
        results = []
        
        # Try each model
        models = [
            ("Logistic Regression", self.evaluate_logistic_regression),
            ("SVM", self.evaluate_svm),
            ("Random Forest", self.evaluate_random_forest),
            ("DistilBERT", self.evaluate_distilbert),
        ]
        
        for name, eval_func in models:
            result = eval_func()
            if result:
                results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{MODELS_DIR}/evaluation_results.csv", index=False)
        logger.info(f"\nEvaluation results saved to {MODELS_DIR}/evaluation_results.csv")
        
        return results_df


def main():
    """Entry point: load test data and evaluate all models."""
    # Load test data
    loader = DataLoader()
    test_df = loader.load("data/test.csv")
    
    # Evaluate
    evaluator = ModelEvaluator(test_df)
    results = evaluator.evaluate_all()
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()