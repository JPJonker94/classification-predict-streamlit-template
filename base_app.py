"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
nltk.download('wordnet')

# functions to clean the data


def remove_urls_lower(word):
    # identify url pattern
    url_pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    # new string for url
    url = r'urlweb'
    # replace with string
    sentence_df = pd.DataFrame([word])
    sentence_df = sentence_df.replace(to_replace=url_pattern, value=url, regex=True)
    return sentence_df.iloc[0][0].lower()


def remove_punctuation(message):
    # identify all characters that should be removed
    punctuation = string.punctuation
    numbers = '0123456789'
    special_char = 'â€¦","ðŸ¥³ã¢‚¬.ï¿½'
    remove_char = punctuation + numbers + special_char
    # remove by using list comprehension
    cleaned_sentence = ''.join([char for char in message if char not in remove_char])
    # Remove unicode characters
    encode = cleaned_sentence.encode('ascii', 'ignore')
    decode_cleaned_sentence = encode.decode()
    return decode_cleaned_sentence


# Instance of lemmatizer
lemma = WordNetLemmatizer()


# lemmatise all of the words
def lemma_df(sentence, lemma):
    spliting = sentence.split(' ')
    lemmatized_sentence = [lemma.lemmatize(word) for word in spliting]
    combine_words = ' '.join(lemmatized_sentence)
    return combine_words

# add all feature names to words
def add_feature_names(vectorized_data):
    df_feature_names = pd.DataFrame(vectorized_data, columns=tweet_cv.get_feature_names())
    return df_feature_names


#create function to change scientific values to float
def change(list_num):
    empty = []
    for x in list_num:
        empty.append(round(float(x),2))
    return empty


#display the location of max probability class and return correct response variable
def locate_max(number_list):
    location_list = number_list.index(max(number_list))
    return location_list-1

# Vectorizer
news_vectorizer = open("resources/CountVectorizer.pkl", "rb")
news_threshold = open("resources/threshold.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file
tweet_thres = joblib.load(news_threshold)  # loading your threshold from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    global tweet_text_lemma
    st.title("Tweet Neural Network Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ['Prediction MLP', "Prediction NNC", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Identify if a person believes in climate changed based on twitter information. "
                    "[2: News about climate change, "
                    "1: Tweet supports belief of climate change, "
                    "0: Neutral, "
                    "-1: Does not believe in climate change]")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the predication page
    if selection == "Prediction MLP":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text Or Sentence", "Type Here")

        if st.button("Classify"):
            # remove urls function
            tweet_text_url = remove_urls_lower(tweet_text)
            # remove punctuation and special char
            tweet_text_pun = remove_punctuation(tweet_text_url)
            # lemmatize the words
            tweet_text_new = lemma_df(tweet_text_pun, lemma)
            # vectorise data
            vect_text = tweet_cv.transform([tweet_text_new]).toarray()
            # add features names for variance threshold selection
            vectorising_fnames_added = add_feature_names(vect_text)
            # implement threshold on data
            vect_thres = tweet_thres.transform(vectorising_fnames_added)

            # Pickle model used due to storage restrictions on Tenser Flow model
            # Try loading in multiple models to give the user a choice
            predictor_mlp = joblib.load(open(os.path.join("resources/MLP_model.pkl"),"rb"))
            prediction = predictor_mlp.predict(vect_thres)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    if selection == "Prediction NNC":
        from tensorflow.keras.models import load_model
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text Or Sentence", "Type Here")

        if st.button("Classify"):
            # remove urls function
            tweet_text_url = remove_urls_lower(tweet_text)
            # remove punctuation and special char
            tweet_text_pun = remove_punctuation(tweet_text_url)
            # lemmatize the words
            tweet_text_new = lemma_df(tweet_text_pun, lemma)
            # vectorise data
            vect_text = tweet_cv.transform([tweet_text_new]).toarray()
            # add features names for variance threshold selection
            vectorising_fnames_added = add_feature_names(vect_text)
            # implement threshold on data
            vect_thres = tweet_thres.transform(vectorising_fnames_added)

            # I have used my tensor model
            # Try loading in multiple models to give the user a choice
            predictor_tensor_flow = load_model("resources/model.h5")
            # reshape data to fit to mode;
            data_reshaped = np.array(vect_thres).reshape(-1, vect_thres.shape[1])
            # fit model
            prediction = predictor_tensor_flow.predict(data_reshaped)[0]
            # change class probabilities to list
            y_pred_df = prediction.tolist()
            # find index location of max probability class and return class number
            y_pred_df_result = locate_max(change(y_pred_df))

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(y_pred_df_result))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()