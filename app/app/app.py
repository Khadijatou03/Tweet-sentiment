import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment en Français",
    page_icon="📊",
    layout="wide"
)

# Initialisation du modèle
def load_model():
    try:
        # Utilisation d'un modèle multilingue qui fonctionne bien avec le français
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        classifier = pipeline("sentiment-analysis", model=model_name)
        return classifier
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def analyze_sentiment(text):
    try:
        classifier = load_model()
        if classifier is None:
            return None, 0
            
        result = classifier(text)
        # Le modèle retourne un score de 1 à 5
        score = int(result[0]['label'].split()[0])
        
        # Conversion du score en sentiment
        if score <= 2:
            sentiment = "Négatif"
            normalized_score = 1 - ((score - 1) / 4)  # Convert 1-2 to high-low negative scores
        elif score == 3:
            sentiment = "Neutre"
            normalized_score = 0.5
        else:
            sentiment = "Positif"
            normalized_score = (score - 2) / 4  # Convert 4-5 to low-high positive scores
            
        return sentiment, normalized_score
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None, 0

# Interface utilisateur
st.title("📊 Analyse de Sentiment en Français")
st.write("Entrez votre texte en français pour analyser son sentiment :")

# Zone de texte pour l'entrée
texte = st.text_area("Texte", height=100)

# Bouton d'analyse
if st.button("Analyser le sentiment"):
    if texte:
        with st.spinner("Analyse en cours..."):
            sentiment, score = analyze_sentiment(texte)
            
            if sentiment is not None:
                # Affichage des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Résultat")
                    st.write(f"Sentiment détecté : **{sentiment}**")
                    st.write(f"Niveau de confiance : {score:.2%}")
                
                with col2:
                    st.subheader("Visualisation")
                    df = pd.DataFrame({
                        'Sentiment': ['Négatif', 'Neutre', 'Positif'],
                        'Score': [
                            score if sentiment == "Négatif" else 0,
                            score if sentiment == "Neutre" else 0,
                            score if sentiment == "Positif" else 0
                        ]
                    })
                    fig = px.bar(df, x='Sentiment', y='Score',
                                color='Sentiment',
                                color_discrete_sequence=['red', 'gray', 'green'])
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")