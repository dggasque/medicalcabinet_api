"""Code for our api app"""
from flask import Flask, jsonify, request
import basilica
import numpy as np
import pandas as pd
from scipy import spatial
app = Flask(__name__)


@app.route('/')
def root():
    return "We have the best API!"

@app.route('/strains', methods=['Post'])
def strains():
    """ a route, expects json object with 1 key """

    # receive input
    lines = request.get_json(force=True)

    # get data from json
    text = lines['input']  # json keys to be determined

    # validate input (optional)
    assert isinstance(text, str)


    # predict
    output = predict(text)


    # give output to sender.
    return output

@app.route('/symptom', methods=['Post'])
def symptom():
    """ a route, expects json object with 1 key """

    # receive input
    lines = request.get_json(force=True)

    # get data from json
    text = lines['input']  # json keys to be determined

    # validate input (optional)
    assert isinstance(text, str)


    # predict
    output = predict_symptoms(text)


    # give output to sender.
    return output

# 4 spaced symptoms json version
# user input
user_input_symp = "multiple sclerosis, epilepsy, pain, "

def predict_symptoms(user_input_symp):

    #unpickling file of embedded cultivar symptoms diseases
    unpickled_df_test = pd.read_pickle("./symptommedembedv6.pkl")

    # getting data
    df = pd.read_csv('symptoms6_medcab3.csv')

    # Part 1
    # a function to calculate_user_text_embedding
    # to save the embedding value in session memory
    user_input_embedding = 0

    def calculate_user_text_embedding(input, user_input_embedding):

        # setting a string of two sentences for the algo to compare
        sentences = [input]

        # calculating embedding for both user_entered_text and for features
        with basilica.Connection('36a370e3-becb-99f5-93a0-a92344e78eab') as c:
            user_input_embedding = list(c.embed_sentences(sentences))
        
        return user_input_embedding

    # run the function to save the embedding value in session memory
    user_input_embedding = calculate_user_text_embedding(user_input, user_input_embedding)




    # part 2
    score = 0

    def score_user_input_from_stored_embedding_from_stored_values(input, score, row1, user_input_embedding):

        # obtains pre-calculated values from a pickled dataframe of arrays
        embedding_stored = unpickled_df_test.loc[row1, 0]
        
        # calculates the similarity of user_text vs. product description
        score = 1 - spatial.distance.cosine(embedding_stored, user_input_embedding)

        # returns a variable that can be used outside of the function
        return score



    # Part 3
    for i in range(2351):
        # calls the function to set the value of 'score'
        # which is the score of the user input
        score = score_user_input_from_stored_embedding_from_stored_values(user_input_symp, score, i, user_input_embedding)
        
        #stores the score in the dataframe
        df.loc[i,'score'] = score


    # Part 4: returns all data for the top 5 results as a json obj
    df_big_json = df.sort_values(by='score', ascending=False)
    df_big_json = df_big_json.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
    df_big_json = df_big_json[:5]
    df_big_json = df_big_json.to_json(orient='columns')

    # Part 5: output
    return df_big_json

# 4 spaced effect json version

# user input
user_input = "text, Relaxed, Violet, Aroused, Creative, Happy, Energetic, Flowery, Diesel"


def predict(user_input):

    # getting data
    df = pd.read_csv('symptoms6_medcab3.csv')

    #effcts unpickling file of embedded cultivar descriptions
    unpickled_df_test = pd.read_pickle("./medembedv2.pkl")

    # Part 1
    # a function to calculate_user_text_embedding
    # to save the embedding value in session memory
    user_input_embedding = 0

    def calculate_user_text_embedding(input, user_input_embedding):

        # setting a string of two sentences for the algo to compare
        sentences = [input]

        # calculating embedding for both user_entered_text and for features
        with basilica.Connection('36a370e3-becb-99f5-93a0-a92344e78eab') as c:
            user_input_embedding = list(c.embed_sentences(sentences))
        
        return user_input_embedding

    # run the function to save the embedding value in session memory
    user_input_embedding = calculate_user_text_embedding(user_input, user_input_embedding)




    # part 2
    score = 0

    def score_user_input_from_stored_embedding_from_stored_values(input, score, row1, user_input_embedding):

        # obtains pre-calculated values from a pickled dataframe of arrays
        embedding_stored = unpickled_df_test.loc[row1, 0]
        
        # calculates the similarity of user_text vs. product description
        score = 1 - spatial.distance.cosine(embedding_stored, user_input_embedding)

        # returns a variable that can be used outside of the function
        return score



    # Part 3
    for i in range(2351):
        # calls the function to set the value of 'score'
        # which is the score of the user input
        score = score_user_input_from_stored_embedding_from_stored_values(user_input, score, i, user_input_embedding)
        
        #stores the score in the dataframe
        df.loc[i,'score'] = score

    # Part 4: returns all data for the top 5 results as a json obj
    df_big_json = df.sort_values(by='score', ascending=False)
    df_big_json = df_big_json.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
    df_big_json = df_big_json[:5]
    df_big_json = df_big_json.to_json(orient='columns')

    # Part 5: output
    return df_big_json