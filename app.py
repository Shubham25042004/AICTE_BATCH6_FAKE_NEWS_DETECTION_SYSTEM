import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK download
nltk.download('stopwords')

# 1. Preprocessing Logic
ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return " ".join(stemmed_content)


# 2. Load the model files

model = joblib.load('fake_news_model.pkl')
vector = joblib.load('tfidf_vectorizer.pkl')

# 3. Streamlit UI Design
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title('🛡️ AICTE Fake News Detection System')
st.subheader('By Shubham Goel')

input_text = st.text_area('Enter the news article content below:', height=200)

if st.button('Predict News Authenticity'):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocessing aur Prediction
        processed_text = stemming(input_text)
        vector_input = vector.transform([processed_text])
        prediction = model.predict(vector_input)

        # Result Display
        if prediction[0] == 1:
            st.error('🚨 Result: This news is likely FAKE!')
        else:
            st.success('✅ Result: This news appears to be REAL.')