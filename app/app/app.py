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

@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_name)

def analyze_sentiment(text):
    classifier = load_model()
    result = classifier(text)
    # Convertir le score 1-5 en catégorie de sentiment
    score = int(result[0]['label'].split()[0])
    if score <= 2:
        return "Négatif", score
    elif score == 3:
        return "Neutre", score
    else:
        return "Positif", score

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
                st.write(f"Score : {score}/5")
            
            with col2:
                st.subheader("Visualisation")
                # Création du graphique
                sentiments = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
                scores = [1 if i == score else 0 for i in range(1, 6)]
                df = pd.DataFrame({
                    'Sentiment': sentiments,
                    'Score': scores
                })
                fig = px.bar(df, x='Sentiment', y='Score',
                            color='Sentiment',
                            color_discrete_sequence=['red', 'salmon', 'gray', 'lightgreen', 'green'])
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)
    else:
        st.warning("Veuillez entrer un texte à analyser.")

# Pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ par l'équipe d'analyse de sentiments")