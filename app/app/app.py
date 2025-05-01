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

@st.cache_resource
def load_model():
    # Utilisation d'un modèle spécifique pour le français
    model_name = "tblard/tf-allocine"  # Modèle français basé sur CamemBERT et entraîné sur des critiques Allociné
    return pipeline("sentiment-analysis", model=model_name)

def analyze_sentiment(text):
    classifier = load_model()
    result = classifier(text)
    # Le modèle retourne 'POSITIVE' ou 'NEGATIVE'
    sentiment = result[0]['label']
    score = result[0]['score']
    
    if sentiment == 'POSITIVE':
        return "Positif", score
    else:
        return "Négatif", score

# Titre de l'application
st.title("📊 Analyse de Sentiment en Français")
st.write("Entrez votre texte en français pour analyser son sentiment :")

# Zone de texte pour l'entrée
texte = st.text_area("Texte", height=100)

# Bouton d'analyse
if st.button("Analyser le sentiment"):
    if texte:
        with st.spinner("Analyse en cours..."):
            sentiment, score = analyze_sentiment(texte)
            
            # Affichage des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Résultat")
                st.write(f"Sentiment détecté : **{sentiment}**")
                st.write(f"Niveau de confiance : {score:.2%}")
            
            with col2:
                st.subheader("Visualisation")
                # Création du graphique
                df = pd.DataFrame({
                    'Sentiment': ['Négatif', 'Positif'],
                    'Score': [1 - score, score] if sentiment == "Positif" else [score, 1 - score]
                })
                fig = px.bar(df, x='Sentiment', y='Score',
                            color='Sentiment',
                            color_discrete_sequence=['red', 'green'])
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")