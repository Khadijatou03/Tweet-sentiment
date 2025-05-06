import streamlit as st
from transformers import pipeline, XLMRobertaTokenizer, AutoModelForSequenceClassification, MBartForConditionalGeneration, MBart50TokenizerFast
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment Multilingue avec Support du Wolof",
    page_icon="📊",
    layout="wide"
)

# Mapping des codes de langue pour MBart
MBART_LANG_CODES = {
    'fr': 'fr_XX',
    'en': 'en_XX',
    'es': 'es_XX',
    'de': 'de_DE',
    'it': 'it_IT',
    'ar': 'ar_AR',
    'wo': 'wo_AF'
}

@st.cache_resource
def load_language_detector():
    """
    Charge le modèle de détection de langue.
    """
    try:
        return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    except Exception as e:
        st.error(f"Erreur lors du chargement du détecteur de langue : {str(e)}")
        return None

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

@st.cache_resource
def load_translator():
    """
    Charge le modèle de traduction multilingue MBart.
    """
    try:
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Erreur lors du chargement du traducteur : {str(e)}")
        return None, None

def detect_language(text):
    """
    Détecte la langue du texte et retourne le code de langue MBart correspondant.
    """
    try:
        detector = load_language_detector()
        if detector is None:
            return 'fr_XX'
            
        result = detector(text)[0]
        detected = result['label'].lower()
        return MBART_LANG_CODES.get(detected, 'fr_XX')
    except:
        return 'fr_XX'  # Par défaut français si la détection échoue

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

def translate_to_wolof(text, source_lang=None):
    """
    Traduit le texte en wolof en utilisant le modèle MBart.
    """
    try:
        model, tokenizer = load_translator()
        if model is None or tokenizer is None:
            return None, None

        # Détection automatique de la langue source si non spécifiée
        if source_lang is None:
            source_lang = detect_language(text)
        
        # Configuration pour le wolof
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Génération de la traduction
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["wo_AF"],
            max_length=512,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.0
        )
        
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation, source_lang
            
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {str(e)}")
        return None, None

# Interface utilisateur
st.title("📊 Analyse de Sentiment Multilingue avec Support du Wolof")
st.write("Écrivez votre texte dans n'importe quelle langue - le système détectera automatiquement la langue!")

# Zone de texte pour l'entrée
texte = st.text_area("Votre texte", height=100)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyser le sentiment"):
        if texte:
            with st.spinner("Analyse en cours..."):
                # Détection de la langue
                langue_detectee = detect_language(texte)
                st.info(f"Langue détectée : {langue_detectee.split('_')[0].upper()}")
                
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
    if texte:
        if st.button("Traduire en Wolof"):
            with st.spinner("Traduction en cours..."):
                traduction, langue_source = translate_to_wolof(texte)
                if traduction:
                    st.subheader("Traduction en Wolof")
                    st.write(f"Traduit depuis : {langue_source.split('_')[0].upper()}")
                    st.write(traduction)
                    
                    # Analyse du sentiment de la traduction
                    sentiment_trad, score_trad = analyze_sentiment(traduction)
                    if sentiment_trad is not None:
                        st.write(f"Sentiment de la traduction : **{sentiment_trad}**")
                        st.write(f"Niveau de confiance : {score_trad:.2%}")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")