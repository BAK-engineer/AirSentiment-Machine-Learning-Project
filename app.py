# MACHINE LEARNING PROJECT IMPLEMENTATION
# AirSentiment Analysis (Airline Tweet Sentiment Predictor)

import streamlit as st
import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# Downloading NLTK data 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Loading model and vectorizer

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function

stop_words = set(stopwords.words("english"))
stop_words.discard("not")   # Preserve 'not' for negation handling

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Keep only alphabets
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # Join words back
    return " ".join(tokens)


# Streamlit App UI

st.set_page_config(page_title="Airline Sentiment", layout="wide")

# Header
st.markdown("<h1 style='text-align: center;'>✈️ Airline Tweet Sentiment Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Check whether a tweet is Positive, Neutral, or Negative</h4>", unsafe_allow_html=True)
st.markdown("---")

# logo
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Airplane_silhouette.svg/600px-Airplane_silhouette.svg.png", width=120)


# Input section
st.subheader("📥 Enter or select a tweet to analyze sentiment")

# User text input
tweet_input = st.text_area("Type your tweet below:")

# Sample tweets dropdown
sample_tweet = st.selectbox(
    "Or try a sample tweet:",
    [
        "I love the flight experience!",
        "The flight was delayed and the staff was rude.",
        "It was okay, nothing special really."
    ]
)

# Use typed tweet if available, otherwise use sample tweet
final_tweet = tweet_input.strip() if tweet_input.strip() != "" else sample_tweet

# Predict button
if st.button("Predict Sentiment"):

    # Clean and vectorize text
    cleaned = clean_text(final_tweet)
    st.write("Cleaned Text:", cleaned)
    vector = vectorizer.transform([cleaned])

    # Predict sentiment
    prediction = model.predict(vector)[0]

    # Display result
    st.subheader("📊📈 Predicted Sentiment:")

    if prediction == "positive":
        st.success("😊 Positive — Great experience!")

    elif prediction == "neutral":
        st.info("😐 Neutral — Average or mixed opinion.")

    else:
        st.error("😡 Negative — Poor experience or complaint.")

# Visualizations from dataset

st.markdown("---")
st.header(" Sentiment Analysis on Full Dataset")

# Loading full dataset
df = pd.read_csv("Tweets.csv")

# Sentiment distribution chart
st.subheader("Sentiment Distribution (original dataset)")
fig, ax = plt.subplots()
sns.countplot(
    data=df,
    x="airline_sentiment",
    hue="airline_sentiment",
    order=["positive", "neutral", "negative"],
    palette="Set2",
    legend=False,
    ax=ax
)
st.pyplot(fig)

# Word Cloud Section
st.subheader(" Word Clouds by Sentiment")

def plot_wordcloud(sentiment):
    text = " ".join(df[df.airline_sentiment == sentiment]["text"].astype(str))
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**😊 Positive**")
    st.pyplot(plot_wordcloud("positive"))

with col2:
    st.markdown("**😐 Neutral**")
    st.pyplot(plot_wordcloud("neutral"))

with col3:
    st.markdown("**😡 Negative**")
    st.pyplot(plot_wordcloud("negative"))


# Infomation Section

with st.expander("ℹ How it works"):
    st.markdown("""
    - This app uses **NLTK** for text preprocessing (stopwords, lemmatization).
    - It uses a **TF-IDF vectorizer** to convert text into numeric format.
    - A **Multinomial Naive Bayes model** predicts whether the tweet is Positive, Neutral, or Negative.
    - Word clouds and charts provide overall insights into public sentiment toward airlines.
    """)


#streamlit run app.py