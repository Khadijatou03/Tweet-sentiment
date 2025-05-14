import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from sklearn.model_selection import train_test_split
import logging
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

def load_and_prepare_data():
    """Charge et prépare les données"""
    logging.info("Chargement des données...")
    
    try:
        # Vérifier l'existence des fichiers de données Wolof
        data_path = "data/dataset_wolof/train_wo.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le fichier {data_path} n'existe pas!")
        
        # Charger les données
        data = pd.read_csv(data_path)
        
        # Nettoyer les données
        data = data.dropna(subset=['text', 'sentiment'])
        data['text'] = data['text'].astype(str)
        
        # Convertir les labels
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        data['labels'] = data['sentiment'].map(label_map)
        
        logging.info(f"Données chargées avec succès. Total échantillons: {len(data)}")
        return data
        
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données: {str(e)}")
        raise

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    try:
        # Configuration
        model_path = "./models/wolof-sentiment-model"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Nettoyer le répertoire si des fichiers existent déjà
        if os.listdir(model_path):
            logging.warning(f"Nettoyage du répertoire {model_path}...")
            for file in os.listdir(model_path):
                os.remove(os.path.join(model_path, file))
        
        # Charger le modèle et tokenizer
        logging.info("Chargement du modèle de base...")
        model_name = "hibaid01/sentiment-analysis-wolof"  # Modèle pré-entraîné pour le Wolof
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Préparer les données
        data = load_and_prepare_data()
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        logging.info(f"Données divisées - Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Créer les datasets
        train_dataset = SentimentDataset(
            texts=train_data['text'].tolist(),
            labels=train_data['labels'].tolist(),
            tokenizer=tokenizer
        )
        
        val_dataset = SentimentDataset(
            texts=val_data['text'].tolist(),
            labels=val_data['labels'].tolist(),
            tokenizer=tokenizer
        )
        
        # Configuration de l'entraînement avec checkpoints
        training_args = TrainingArguments(
            output_dir=model_path,
            num_train_epochs=5,  # Plus d'époques pour un meilleur apprentissage
            per_device_train_batch_size=8,  # Batch size plus petit pour éviter les OOM
            per_device_eval_batch_size=8,
            evaluation_strategy="steps",
            eval_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(model_path, 'logs'),
            logging_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            learning_rate=1e-5,  # Learning rate plus faible pour un fine-tuning plus doux
            warmup_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Créer le trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Entraînement
        logging.info("Début du fine-tuning...")
        trainer.train()
        
        # Sauvegarder le modèle final
        logging.info("Sauvegarde du modèle final...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Vérifier la sauvegarde
        required_files = [
            "pytorch_model.bin",
            "config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            raise FileNotFoundError(f"Fichiers manquants après la sauvegarde: {', '.join(missing_files)}")
        
        logging.info("✓ Modèle sauvegardé avec succès!")
        logging.info(f"✓ Emplacement du modèle: {os.path.abspath(model_path)}")
        
    except KeyboardInterrupt:
        logging.warning("\nEntraînement interrompu par l'utilisateur. Tentative de sauvegarde du dernier état...")
        try:
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logging.info("État intermédiaire sauvegardé avec succès!")
        except Exception as e:
            logging.error(f"Échec de la sauvegarde de l'état intermédiaire: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise

if __name__ == "__main__":
    train()
