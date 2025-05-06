import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment et Traduction Wolof",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Charge un modèle d'analyse de sentiment robuste et multilingue.
    
    Retourne un objet `pipeline` de Hugging Face permettant d'analyser le sentiment
    d'un texte en utilisant le modèle "nlptown/bert-base-multilingual-uncased-sentiment".
    
    Si le modèle n'est pas disponible (par exemple si le modèle n'a pas encore été
    téléchargé), affiche un message d'erreur et retourne `None`.
    """
    try:
        # Utilisation d'un modèle multilingue plus stable
        return pipeline(
            task="sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

def analyze_sentiment(text):
    try:
        classifier = load_model()
        if classifier is None:
            return None, 0
            
        result = classifier(text)
        score = result[0]['score']
        rating = int(result[0]['label'].split()[0])  # Extraire le nombre de 1 à 5
        
        # Conversion du rating 1-5 en sentiment
        if rating <= 2:
            sentiment = 'Négatif'
        elif rating == 3:
            sentiment = 'Neutre'
        else:
            sentiment = 'Positif'
        
        return sentiment, score
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None, 0

@st.cache_resource
def load_translator():
    """
    Charge le modèle de traduction français-wolof.
    """
    try:
        return pipeline(
            "translation",
            model="bilalfaye/nllb-200-distilled-600M-wolof-french",
            src_lang="fra_Latn",
            tgt_lang="wol_Latn"
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du traducteur : {str(e)}")
        return None

def translate_text(text, source_lang="fra_Latn", target_lang="wol_Latn"):
    """
    Traduit le texte entre le français et le wolof.
    """
    try:
        translator = load_translator()
        if translator is None:
            return None
            
        result = translator(text, src_lang=source_lang, tgt_lang=target_lang)
        return result[0]['translation_text']
            
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {str(e)}")
        return None

# Interface utilisateur
st.title("📊 Analyse de Sentiment et Traduction Wolof")
st.write("Entrez votre texte en français pour l'analyser et le traduire en wolof :")

# Zone de texte pour l'entrée
texte = st.text_area("Texte", height=100)

col1, col2 = st.columns(2)

with col1:
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

with col2:
    if st.button("Traduire en Wolof"):
        if texte:
            with st.spinner("Traduction en cours..."):
                traduction = translate_text(texte)
                if traduction:
                    st.subheader("Traduction en Wolof")
                    st.write(traduction)
                    
                    # Option pour traduire du Wolof vers le Français
                    if st.button("Retraduire vers le Français"):
                        retraduction = translate_text(traduction, 
                                                   source_lang="wol_Latn", 
                                                   target_lang="fra_Latn")
                        if retraduction:
                            st.subheader("Retraduction en Français")
                            st.write(retraduction)
        else:
            st.warning("Veuillez entrer un texte à traduire.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")