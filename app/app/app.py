import streamlit as st
from transformers import pipeline, XLMRobertaTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import plotly.express as px
import pandas as pd

# Configuration des langues supportées
SUPPORTED_LANGUAGES = {
    "Français": "fr",
    "English": "en",
    "Español": "es",
    "Deutsch": "de",
    "Italiano": "it",
    "Wolof": "wo"  # Ajout du Wolof
}

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
        
        # Utilisation explicite de XLMRobertaTokenizer
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

@st.cache_resource
def load_translator():
    """
    Charge le modèle de traduction.
    """
    try:
        return pipeline(
            "translation",
            model="t5-base",
            src_lang="fr_Latn",
            tgt_lang="wo_Latn"
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du traducteur : {str(e)}")
        return None

def translate_text(text, source_lang="fr_Latn", target_lang="wo_Latn"):
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
st.title("📊 Analyse de Sentiment Multilingue")
st.write("Analysez vos textes dans différentes langues :")

# Sélection de la langue
langue = st.selectbox(
    "Langue du texte",
    options=list(SUPPORTED_LANGUAGES.keys())
)

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
    # Sélection de la langue cible pour la traduction
    target_lang = st.selectbox(
        "Traduire vers",
        options=[lang for lang in SUPPORTED_LANGUAGES.keys() if lang != langue]
    )
    
    if st.button(f"Traduire en {target_lang}"):
        if texte:
            with st.spinner("Traduction en cours..."):
                source_lang_code = f"{SUPPORTED_LANGUAGES[langue]}_Latn"
                target_lang_code = f"{SUPPORTED_LANGUAGES[target_lang]}_Latn"
                
                traduction = translate_text(texte, 
                                         source_lang=source_lang_code,
                                         target_lang=target_lang_code)
                if traduction:
                    st.subheader(f"Traduction en {target_lang}")
                    st.write(traduction)
                    
                    # Analyse du sentiment de la traduction
                    if st.button(f"Analyser le sentiment de la traduction"):
                        sentiment_trad, score_trad = analyze_sentiment(traduction)
                        if sentiment_trad is not None:
                            st.write(f"Sentiment détecté : **{sentiment_trad}**")
                            st.write(f"Niveau de confiance : {score_trad:.2%}")
        else:
            st.warning("Veuillez entrer un texte à traduire.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")