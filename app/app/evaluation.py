import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline, XLMRobertaTokenizer, AutoModelForSequenceClassification

class ModelEvaluator:
    def __init__(self):
        self.base_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.fine_tuned_model_path = "./models/twitter-sentiment-model"
        
    def load_model(self, use_fine_tuned=False):
        """Charge le modèle approprié"""
        model_path = self.fine_tuned_model_path if use_fine_tuned else self.base_model_name
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {str(e)}")
            return None

    def evaluate_dataset(self, texts, true_labels, use_fine_tuned=False):
        """Évalue le modèle sur un dataset"""
        classifier = self.load_model(use_fine_tuned)
        if classifier is None:
            return None
        
        predictions = []
        for text in texts:
            try:
                result = classifier(text)[0]
                pred_label = result['label'].lower()
                predictions.append(pred_label)
            except Exception as e:
                print(f"Erreur lors de la prédiction : {str(e)}")
                predictions.append("neutral")  # valeur par défaut en cas d'erreur
        
        # Calcul des métriques
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }

    def plot_confusion_matrix(self, conf_matrix, title):
        """Affiche la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('Vraies étiquettes')
        plt.xlabel('Prédictions')
        plt.show()

def main():
    # Chargement des données de test
    # Note: Remplacez ces chemins par vos propres fichiers de test
    test_en = pd.read_csv("./data/test_english.csv")
    test_fr = pd.read_csv("./data/test_french.csv")
    
    evaluator = ModelEvaluator()
    
    print("=== Évaluation avant fine-tuning ===")
    
    # Test sur données anglaises
    print("\nTest sur données anglaises (avant fine-tuning):")
    results_en = evaluator.evaluate_dataset(
        test_en['text'].tolist(),
        test_en['label'].tolist(),
        use_fine_tuned=False
    )
    if results_en:
        print(f"Accuracy: {results_en['accuracy']:.4f}")
        print("\nRapport de classification:")
        print(results_en['classification_report'])
        evaluator.plot_confusion_matrix(
            results_en['confusion_matrix'],
            "Matrice de confusion - Anglais (avant fine-tuning)"
        )
    
    # Test sur données françaises
    print("\nTest sur données françaises (avant fine-tuning):")
    results_fr = evaluator.evaluate_dataset(
        test_fr['text'].tolist(),
        test_fr['label'].tolist(),
        use_fine_tuned=False
    )
    if results_fr:
        print(f"Accuracy: {results_fr['accuracy']:.4f}")
        print("\nRapport de classification:")
        print(results_fr['classification_report'])
        evaluator.plot_confusion_matrix(
            results_fr['confusion_matrix'],
            "Matrice de confusion - Français (avant fine-tuning)"
        )
    
    print("\n=== Évaluation après fine-tuning ===")
    
    # Test sur données anglaises
    print("\nTest sur données anglaises (après fine-tuning):")
    results_en_ft = evaluator.evaluate_dataset(
        test_en['text'].tolist(),
        test_en['label'].tolist(),
        use_fine_tuned=True
    )
    if results_en_ft:
        print(f"Accuracy: {results_en_ft['accuracy']:.4f}")
        print("\nRapport de classification:")
        print(results_en_ft['classification_report'])
        evaluator.plot_confusion_matrix(
            results_en_ft['confusion_matrix'],
            "Matrice de confusion - Anglais (après fine-tuning)"
        )
    
    # Test sur données françaises
    print("\nTest sur données françaises (après fine-tuning):")
    results_fr_ft = evaluator.evaluate_dataset(
        test_fr['text'].tolist(),
        test_fr['label'].tolist(),
        use_fine_tuned=True
    )
    if results_fr_ft:
        print(f"Accuracy: {results_fr_ft['accuracy']:.4f}")
        print("\nRapport de classification:")
        print(results_fr_ft['classification_report'])
        evaluator.plot_confusion_matrix(
            results_fr_ft['confusion_matrix'],
            "Matrice de confusion - Français (après fine-tuning)"
        )

if __name__ == "__main__":
    main()
