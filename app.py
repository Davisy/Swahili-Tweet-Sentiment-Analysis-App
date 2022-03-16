import streamlit as st
import requests as r
from os.path import dirname, join, realpath
import joblib
from langdetect import detect

# add banner image
st.header("Swahili Tweet Sentiment Analysis App")
st.image("images/sentiment-image.png")
st.subheader(
    """
A simple app to analyze the sentiment of Swahili tweets.
"""
)

# form to collect news content
my_form = st.form(key="news_form")
tweet = my_form.text_input("Input your swahili tweet here")
submit = my_form.form_submit_button(label="make prediction")


# load the model and count_vectorizer

with open(
    join(dirname(realpath(__file__)), "model/swahili-sentiment-model.pkl"), "rb"
) as f:
    model = joblib.load(f)

with open(join(dirname(realpath(__file__)), "preprocessing/vectorizer.pkl"), "rb") as f:
    vectorizer = joblib.load(f)

sentiments = {0: "Neutral", 1: "Positive", -1: "Negative"}


if submit:

    if detect(tweet) == "sw":

        # transform the input
        transformed_tweet = vectorizer.transform([tweet])

        # perform prediction
        prediction = model.predict(transformed_tweet)
        output = int(prediction[0])
        probas = model.predict_proba(transformed_tweet)
        probability = "{:.2f}".format(float(probas[:, output]))

        # Display results of the NLP task
        st.header("Results")
        if output == 1:
            st.write("The sentiment of the tweet is {} 😊".format(sentiments[output]))
        elif output == -1:
            st.write("The sentiment of the tweet is {} 😡 ".format(sentiments[output]))
        else:
            st.write("The sentiment of the tweet is {} 😐".format(sentiments[output]))

    else:
        st.write(" ⚠️ The tweet is not in swahili language.Please make sure the input is in swahili language")

url = "https://twitter.com/Davis_McDavid"
st.write("Developed with ❤️ by [Davis David](%s)" % url)
