import re
import pandas as pd
import numpy as np

class CustomNaiveBayesClassifier:
    """
    Custom Naive Bayes Text Classifier - recreated to fix pickle loading
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {}
        self.vocabulary = set()
        self.classes = []
        self.trained = False
        self.training_stats = {}
        
    def preprocess_text(self, text):
        """Preprocess text for training/prediction"""
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase and remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        
        # Split and filter words (keep words with length > 2)
        words = [word for word in text.split() if len(word) > 2]
        
        return words
    
    def predict(self, X):
        """Predict classes for documents"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        for doc in X:
            pred = self._predict_single(doc)
            predictions.append(pred)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        probabilities = []
        for doc in X:
            proba = self._predict_proba_single(doc)
            probabilities.append(proba)
        
        return probabilities
    
    def _predict_single(self, document):
        """Predict class for a single document"""
        words = self.preprocess_text(document)
        log_probs = {}
        
        for cls in self.classes:
            log_prob = np.log(self.class_probs[cls])
            
            for word in words:
                if word in self.vocabulary:
                    log_prob += np.log(self.word_probs[cls][word])
                else:
                    # Handle unknown words
                    log_prob += np.log(self.alpha / (len(self.vocabulary) + self.alpha))
            
            log_probs[cls] = log_prob
        
        return max(log_probs, key=log_probs.get)
    
    def _predict_proba_single(self, document):
        """Predict probabilities for a single document"""
        words = self.preprocess_text(document)
        log_probs = {}
        
        for cls in self.classes:
            log_prob = np.log(self.class_probs[cls])
            for word in words:
                if word in self.vocabulary:
                    log_prob += np.log(self.word_probs[cls][word])
                else:
                    log_prob += np.log(self.alpha / (len(self.vocabulary) + self.alpha))
            log_probs[cls] = log_prob
        
        # Convert to probabilities using softmax
        max_log_prob = max(log_probs.values())
        exp_probs = {cls: np.exp(log_prob - max_log_prob) for cls, log_prob in log_probs.items()}
        total_exp = sum(exp_probs.values())
        
        return {cls: prob / total_exp for cls, prob in exp_probs.items()}