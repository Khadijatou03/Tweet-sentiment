import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
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
        # Utilisation de CamemBERT pour le français
        model_name = "camembert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained("almanach/camembert-base-sentiment")
        
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
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
        label = result[0]['label']
        score = result[0]['score']
        
        # Conversion des labels
        sentiment_map = {
            'POSITIVE': 'Positif',
            'NEGATIVE': 'Négatif',
            'NEUTRAL': 'Neutre'
        }
        
        sentiment = sentiment_map.get(label, 'Neutre')
        return sentiment, score
        
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