import joblib
import numpy as np
import re
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk_resources = ['stopwords', 'wordnet']

for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class RestaurantReviewPredictor:
    def __init__(self, model_path='restaurant_review_classifier.joblib'):
        """
        Initialize the predictor with a trained model
        """
        self.pipeline = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load the model
        self.load_model(model_path)
    
    def preprocess_text(self, text):
        """
        Preprocess text for prediction (same as training)
        """
        if not text or text.strip() == "":
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords, then lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_model(self, model_path):
        """
        Load the trained model from joblib file
        """
        try:
            model_data = joblib.load(model_path)
            
            self.pipeline = model_data['pipeline']
            self.label_mapping = model_data['label_mapping']
            self.reverse_label_mapping = model_data['reverse_label_mapping']
            
        except FileNotFoundError:
            print(f"Error: '{model_path}' not found!")
            print("Please make sure you have trained a model first.")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def predict(self, text, show_all_probabilities=False):
        """
        Predict the classification of input text
        
        Args:
            text (str): Input text to classify
            show_all_probabilities (bool): Whether to return probabilities for all classes
            
        Returns:
            dict: Prediction results with label, confidence, and optionally all probabilities
        """
        if not text or text.strip() == "":
            return {
                'error': 'Empty text provided',
                'prediction': None,
                'confidence': 0.0
            }
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Make prediction
            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            
            # Convert back to original label
            predicted_label = self.reverse_label_mapping[prediction]
            confidence = np.max(probabilities)
            
            result = {
                'prediction': predicted_label,
                'confidence': round(confidence, 4),
                'text_length': len(text),
                'processed_text': processed_text
            }
            
            if show_all_probabilities:
                all_probs = {}
                for i, prob in enumerate(probabilities):
                    label = self.reverse_label_mapping[i]
                    all_probs[label] = round(prob, 4)
                
                # Sort by probability (highest first)
                result['all_probabilities'] = dict(sorted(all_probs.items(), 
                                                         key=lambda x: x[1], 
                                                         reverse=True))
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, texts):
        """
        Predict classifications for multiple texts
        
        Args:
            texts (list): List of text strings to classify
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}...")
            result = self.predict(text, show_all_probabilities=True)
            result['index'] = i
            result['original_text'] = text
            results.append(result)
        
        return results

def nlp(text):
    """
    Main function for interactive prediction
    """

    # Initialize predictor
    try:
        predictor = RestaurantReviewPredictor()
    except Exception as e:
        print("Failed to load model. Exiting.")
        return


    result = predictor.predict(text.strip(), show_all_probabilities=True)
    return (result, text)
