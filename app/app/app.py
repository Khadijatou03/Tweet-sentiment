import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment en Français",
    page_icon="📊",
    layout="wide"
)

# Initialisation du modèle (sans cache pour éviter les erreurs)
def load_model():
    try:
        # Utilisation d'un modèle plus léger spécifique au français
        model_name = "cmarkea/distilcamembert-base-sentiment"
        classifier = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
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
        sentiment = result[0]['label']
        score = result[0]['score']
        
        # Conversion des labels en français
        if sentiment == 'POSITIVE':
            return "Positif", score
        elif sentiment == 'NEGATIVE':
            return "Négatif", score
        else:
            return "Neutre", score
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