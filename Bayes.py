import pickle
import joblib
import os
from datetime import datetime
from collections import defaultdict, Counter
from BayesClass import CustomNaiveBayesClassifier


class RobustModelRunner:
    """
    Robust model runner with better error handling
    """
    
    def __init__(self, model_dir='.'):
        """
        Initialize with the directory containing your model files
        """
        self.model_dir = model_dir
        self.custom_model = None
        self.sklearn_model = None
        self.vectorizer = None
        self.metadata = None
        self.load_models()
    
    def load_models(self):
        """Load all model files with robust error handling"""
        # Load custom Naive Bayes model with error handling
        custom_path = os.path.join(self.model_dir, 'custom_naive_bayes_model.pkl')
        try:
            with open(custom_path, 'rb') as f:
                self.custom_model = pickle.load(f)
        except Exception as e:
            print(f"Error (Custom Model): {e}")
            self.custom_model = None

        # Load sklearn model
        sklearn_path = os.path.join(self.model_dir, 'sklearn_naive_bayes_model.joblib')
        try:
            self.sklearn_model = joblib.load(sklearn_path)
        except Exception as e:
            print(f"Error(SKLearn Model): {e}")
            self.sklearn_model = None
        
        # Load TF-IDF vectorizer
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        try:
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            print(f"Error (Vectorizer): {e}")
            self.vectorizer = None
    
    def predict(self, text):
        """
        Predict classification for a single text using available models
        """
        
        results = {
            'text': text,
            'custom_model': {},
            'sklearn_model': {}
        }
        
        # Custom model prediction
        if self.custom_model:
            custom_pred = self.custom_model.predict([text])[0]
            custom_proba = self.custom_model.predict_proba([text])[0]
            
            results['custom_model'] = {
                'prediction': custom_pred,
                'probabilities': custom_proba,
                'status': 'success'
            }
            
        # Sklearn model prediction
        if self.sklearn_model and self.vectorizer:
            # Transform text using TF-IDF vectorizer
            text_tfidf = self.vectorizer.transform([text])
            sklearn_pred = self.sklearn_model.predict(text_tfidf)[0]
            sklearn_proba = self.sklearn_model.predict_proba(text_tfidf)[0]
            
            # Create probability dictionary
            proba_dict = dict(zip(self.sklearn_model.classes_, sklearn_proba))
            
            results['sklearn_model'] = {
                'prediction': sklearn_pred,
                'probabilities': proba_dict,
                'status': 'success'
            }   
        
        # Show agreement/disagreement if both models worked
        custom_pred = results['custom_model'].get('prediction')
        sklearn_pred = results['sklearn_model'].get('prediction')
        
        return results

def Bayes(text):
    # Initialize runner (assumes model files are in current directory)
    runner = RobustModelRunner('.')
    
    return runner.predict(text)
