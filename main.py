# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) +3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    if len(review)==0:
        st.error('Please enter something in review.', icon="🚨")
        return "No Review", 0
    preprocessed_text=preprocess_text(review)
    prediction = model.predict(preprocessed_text)[0][0]
    sentiment = 'Postive' if prediction>0.5 else 'Negative'
    return sentiment,prediction

st.title("IMDB sentiment prediction on movie review")
st.write("This model classifies the review as Negative or Postive based on user input.")
review=st.text_area("Please enter review below")

if st.button('Classify'):
    sentiment,prediction=predict_sentiment(review)
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Predection : {prediction:.2f}')
else:
    st.write("Please entere a movie review")


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Based on IMDB Dataset<br> Developed with ❤ by <a style='display: block;color: white;text-align: center;' href="https://github.com/Satnamix" target="_blank">Satnam Singh</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
