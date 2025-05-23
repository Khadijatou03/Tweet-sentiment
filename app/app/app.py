import streamlit as st
from transformers import pipeline, XLMRobertaTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment Multilingue",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Charge le modèle XLM-RoBERTa pour l'analyse de sentiment multilingue.
    """
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def analyze_sentiment(text):
    """
    Analyse le sentiment d'un texte et retourne le sentiment et le score.
    """
    try:
        classifier = load_model()
        if classifier is None:
            return None, None
            
        result = classifier(text)[0]
        label = result['label']
        score = result['score']
        
        # Conversion des étiquettes en format lisible
        sentiment_map = {
            'positive': 'Positif',
            'negative': 'Négatif',
            'neutral': 'Neutre'
        }
        
        sentiment = sentiment_map.get(label.lower(), 'Neutre')
        return sentiment, score
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None, None

# Interface utilisateur
st.title("📊 Analyse de Sentiment Multilingue")
st.write("Écrivez votre texte dans n'importe quelle langue (français, anglais, wolof, etc.)")

# Exemples en différentes langues
with st.expander("Voir des exemples de textes"):
    st.write("""
    **Français**: "Je suis très content aujourd'hui!"
    **English**: "This is a great day!"
    **Wolof**: "Dama bëgg lii!"
    """)

# Zone de texte pour l'entrée
texte = st.text_area("Votre texte", height=100)

if st.button("Analyser le sentiment"):
    if texte:
        with st.spinner("Analyse en cours..."):
            sentiment, score = analyze_sentiment(texte)
            
            if sentiment is not None:
                st.subheader("Résultat de l'analyse")
                st.write(f"Sentiment détecté : **{sentiment}**")
                st.write(f"Niveau de confiance : {score:.2%}")
                
                # Visualisation
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