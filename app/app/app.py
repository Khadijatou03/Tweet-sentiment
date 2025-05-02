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
        # Utilisation d'un modèle spécifique pour le français
        model_name = "cmarkea/distilcamembert-base-sentiment"
        classifier = pipeline("sentiment-analysis", model=model_name)
        return classifier
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def analyze_sentiment(text):
    try:
        classifier = load_model()
        if classifier is None:
            return None, None, None
            
        result = classifier(text)
        
        # Le modèle retourne directement les probabilités pour chaque classe
        probs = result[0]
        
        # Déterminer le sentiment avec le score le plus élevé
        if probs['label'] == 'POSITIVE':
            sentiment = "Positif"
        elif probs['label'] == 'NEGATIVE':
            sentiment = "Négatif"
        else:
            sentiment = "Neutre"
            
        # Calculer les probabilités pour chaque classe
        scores = {
            'Positif': probs['score'] if probs['label'] == 'POSITIVE' else 0.1,
            'Neutre': probs['score'] if probs['label'] == 'NEUTRAL' else 0.1,
            'Négatif': probs['score'] if probs['label'] == 'NEGATIVE' else 0.1
        }
            
        return sentiment, probs['score'], scores
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None, None, None

# Interface utilisateur
st.title("📊 Analyse de Sentiment en Français")
st.write("Entrez votre texte en français pour analyser son sentiment :")

# Zone de texte pour l'entrée
texte = st.text_area("Texte", height=100)

# Bouton d'analyse
if st.button("Analyser le sentiment"):
    if texte:
        with st.spinner("Analyse en cours..."):
            sentiment, score, scores = analyze_sentiment(texte)
            
            if sentiment is not None:
                # Affichage des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Résultats")
                    st.write(f"Sentiment prédit : **{sentiment}**")
                    st.write("Probabilités :")
                    for sent, prob in scores.items():
                        st.write(f"- {sent}: {prob:.2%}")
                
                with col2:
                    st.subheader("Visualisation")
                    df = pd.DataFrame({
                        'Sentiment': list(scores.keys()),
                        'Probabilité': list(scores.values())
                    })
                    fig = px.bar(df, x='Sentiment', y='Probabilité',
                                color='Sentiment',
                                color_discrete_sequence=['green', 'gray', 'red'])
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")