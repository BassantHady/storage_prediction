"""
predict.py
==========
Inference script for making predictions using trained models.
Supports single sentences or batch prediction with all available models.

Author: NLP Engineering Team
"""

import pickle
import logging
import numpy as np
import torch
from typing import List, Union, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change from "from src.preprocessing" to "from preprocessing"
from preprocessing import TextPreprocessor, StorageLabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["freezer", "fridge", "normal"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}


class StoragePredictor:
    """
    Unified predictor for all trained models.
    Loads models lazily and provides a simple predict() interface.
    """
    
    def __init__(self, model_name: str = "distilbert"):
        """
        Initialize predictor with a specific model.
        
        Args:
            model_name: One of 'logistic', 'svm', 'rf', 'distilbert'
        """
        self.model_name = model_name.lower()
        self.preprocessor = TextPreprocessor()
        self.label_enc = StorageLabelEncoder()
        self._model = None
        self._tfidf = None
        self._vocab = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the specified model from disk."""
        logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == "logistic":
            with open(f"{MODELS_DIR}/logistic_regression.pkl", "rb") as f:
                self._model = pickle.load(f)
            self._load_tfidf()
            
        elif self.model_name == "svm":
            with open(f"{MODELS_DIR}/svm.pkl", "rb") as f:
                self._model = pickle.load(f)
            self._load_tfidf()
            
        elif self.model_name == "rf":
            with open(f"{MODELS_DIR}/random_forest.pkl", "rb") as f:
                self._model = pickle.load(f)
            self._load_tfidf()
            
        elif self.model_name == "distilbert":
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(f"{MODELS_DIR}/distilbert_tokenizer")
            self._model = DistilBertForSequenceClassification.from_pretrained(f"{MODELS_DIR}/distilbert_best").to(DEVICE)
            self._model.eval()
            self._max_len = 128
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Choose from: logistic, svm, rf, lstm, distilbert")
        
        logger.info(f"Model loaded successfully.")
    
    def _load_tfidf(self):
        """Load TF-IDF vectorizer for traditional models."""
        with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
            self._tfidf = pickle.load(f)
    
    def predict(self, text: str) -> Dict:
        """
        Predict storage type for a single sentence.
        
        Args:
            text: Input sentence asking about storage (e.g., "Should I store milk in the fridge?")
        
        Returns:
            Dictionary with predicted label and confidence scores
        """
        results = self._predict_batch([text])
        return results[0]
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict storage types for multiple sentences.
        
        Args:
            texts: List of input sentences
        
        Returns:
            List of dictionaries with predictions for each text
        """
        return self._predict_batch(texts)
    
    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        """Internal batch prediction method."""
        if self.model_name in ["logistic", "svm", "rf"]:
            return self._predict_traditional(texts)
        elif self.model_name == "distilbert":
            return self._predict_distilbert(texts)
    
    def _predict_traditional(self, texts: List[str]) -> List[Dict]:
        """Prediction for traditional ML models (TF-IDF based)."""
        cleaned_texts = self.preprocessor.transform(texts)
        X = self._tfidf.transform(cleaned_texts)
        pred_ids = self._model.predict(X)
        
        # Get decision function or probabilities if available
        confidence_scores = []
        if hasattr(self._model, "predict_proba"):
            # For Logistic Regression and Random Forest
            probs = self._model.predict_proba(X)
            for i in range(len(pred_ids)):
                confidence_scores.append(float(max(probs[i])))
        elif hasattr(self._model, "decision_function"):
            # For SVM (LinearSVC)
            decision = self._model.decision_function(X)
            # Convert decision scores to pseudo-probabilities using sigmoid
            if len(decision.shape) == 1:
                # Binary decision, but we have 3 classes
                # For multi-class, decision shape is (n_samples, n_classes)
                pass
            # For multi-class LinearSVC
            if len(decision.shape) == 2:
                # Apply softmax to convert to probabilities
                exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probs = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
                for i in range(len(pred_ids)):
                    confidence_scores.append(float(max(probs[i])))
            else:
                confidence_scores.append(None)
        else:
            confidence_scores = [None] * len(pred_ids)
        
        results = []
        for i, pred_id in enumerate(pred_ids):
            result = {
                "text": texts[i],
                "predicted_storage": ID_TO_LABEL[pred_id],
                "confidence": confidence_scores[i] if confidence_scores else None
            }
            results.append(result)
        
        return results
    
    def _predict_distilbert(self, texts: List[str]) -> List[Dict]:
        """Prediction for DistilBERT model."""
        encodings = self._tokenizer(
            texts,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self._model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            pred_ids = outputs.logits.argmax(dim=1).cpu().numpy()
        
        results = []
        for i, pred_id in enumerate(pred_ids):
            result = {
                "text": texts[i],
                "predicted_storage": ID_TO_LABEL[pred_id],
                "confidence": float(probs[i][pred_id]),
                "all_probs": {
                    "freezer": float(probs[i][0]),
                    "fridge": float(probs[i][1]),
                    "normal": float(probs[i][2])
                }
            }
            results.append(result)
        
        return results


def predict_from_text(text: str, model: str = "distilbert") -> Dict:
    """
    Convenience function for quick predictions.
    
    Args:
        text: Input sentence
        model: Model to use ('logistic', 'svm', 'rf', 'lstm', 'distilbert')
    
    Returns:
        Prediction dictionary
    """
    predictor = StoragePredictor(model_name=model)
    return predictor.predict(text)


def main():
    """Demo prediction functionality."""
    # Test sentences
    test_sentences = [
        "Should I store milk in the fridge?",
        "Where should I keep frozen chicken nuggets?",
        "Can I leave my laptop at room temperature?",
    ]
    
    # Use DistilBERT (best model typically)
    predictor = StoragePredictor(model_name="distilbert")
    
    print("\n" + "="*70)
    print("STORAGE TYPE PREDICTIONS")
    print("="*70)
    
    for sentence in test_sentences:
        result = predictor.predict(sentence)
        print(f"\nInput: {result['text']}")
        print(f"Prediction: {result['predicted_storage']}")
        print(f"Confidence: {result['confidence']:.4f}")
        if 'all_probs' in result:
            print(f"All probabilities: {result['all_probs']}")


if __name__ == "__main__":
    main()