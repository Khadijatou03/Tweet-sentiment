import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        # Utilisation du modèle spécifique pour le Wolof
        self.model_name = "hibaid01/sentiment-analysis-wolof"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle Wolof: {str(e)}")
            print("Utilisation du modèle de repli")
            # Utilisation d'un modèle multilingue plus adapté aux langues africaines
            self.model_name = "Davlan/distilbert-base-multilingual-cased-masakhaner"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
            self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
    def preprocess_text(self, text):
        # Nettoyage du texte tout en préservant les caractères spéciaux du Wolof
        text = str(text)
        return text
    
    def predict(self, text):
        # Prétraitement
        text = self.preprocess_text(text)
        
        try:
            # Utilisation du pipeline pour la prédiction
            result = self.classifier(text)[0]
            
            # Mapping du sentiment selon les labels du modèle Wolof
            label = result['label'].upper()
            score = result['score']
            
            # Conversion du résultat au format attendu
            sentiment_map = {
                'POSITIVE': 'Positif',
                'NEGATIVE': 'Négatif',
                'NEUTRAL': 'Neutre'
            }
            
            sentiment = sentiment_map.get(label, 'Neutre')
            
            # Construction des probabilités
            probs = {
                'Positif': score if label == 'POSITIVE' else (1 - score) / 2,
                'Négatif': score if label == 'NEGATIVE' else (1 - score) / 2,
                'Neutre': score if label == 'NEUTRAL' else (1 - score) / 2
            }
            
            return {
                "sentiment": sentiment,
                "probabilities": probs
            }
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            return {
                "sentiment": "Erreur",
                "probabilities": {
                    "Négatif": 0.0,
                    "Neutre": 1.0,
                    "Positif": 0.0
                }
            }