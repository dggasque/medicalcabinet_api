"""Code for our api app"""
from flask import Flask, jsonify, request

app = Flask(__name__)

user_input = "text, Relaxed, Violet, Aroused, Creative, Happy, Energetic, Flowery, Diesel"

# ominbus function
def predict(user_input):

    # install basilica
    #!pip install basilica

    import basilica
    import numpy as np
    import pandas as pd
    from scipy import spatial

    # get data
    #!wget https://raw.githubusercontent.com/MedCabinet/ML_Machine_Learning_Files/master/med1.csv
    # turn data into dataframe
    df = pd.read_csv('med1.csv')

    # get pickled trained embeddings for med cultivars
    #!wget https://github.com/lineality/4.4_Build_files/raw/master/medembedv2.pkl
    #unpickling file of embedded cultivar descriptions
    unpickled_df_test = pd.read_pickle("./medembedv2.pkl")



    # Part 1
    # maybe make a function to perform the last few steps

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

    # Part 4
    #output = df['Strain'].groupby(df['score']).value_counts().nlargest(6, keep='last')
    #output = df['Strain'].groupby(df['score']).value_counts().nlargest(6, keep='last')

    #df_results = df['Strain'].groupby(df['score']).value_counts().nlargest(6, keep='last')
    just_id_output = df['score'].sort_values(ascending=False)

    just_id_output = just_id_output.head(5)

    #print(output)
    #print(output[1:])
    #output_string = str(output)
    #print(type(output))
    #print(output.shape)
    #print(output_string)
    #output_regex = re.sub(r'[^a-zA-Z ^0-9 ^.]', '', output_string)
    #print(output_regex)
    #output_string_clipped = output_regex[50:-28]
    #print(output_string_clipped)

    # json output
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_json.html
    #output_json = output.to_json(orient='index')
    #print(output_json)
    
    output_ID = just_id_output.to_json(orient='index')
    #print(output_ID)

    #output_json = just_id_output.to_json(orient='split')
    #print(output_json)

    #output_values_series = pd.Series(output_json)
    #print(output_values_series)

    #output_json = output_string_clipped.to_json(orient='split')
    #print(output_json)

    #print(type(output_json))
    #print(*output_json)
    #output_json = output.to_json(orient='records')
    #print(output_json)
    #output_json = output.to_json(orient='columns')
    #print(output_json)
    #output_json = output.to_json(orient='values')
    #print(output_json)

    return output_ID

@app.route('/strains', methods=['Post'])
def strains():
    """ a route, expects json object with 1 key """

    # receive input
    lines = request.get_json(force=True)

    # get data from json
    text = lines['input']  # json keys to be determined

    # validate input (optional)
    assert isinstance(text, str)

    # deserialize the pretrained model
    #with open('medembedv2.pkl', 'rb') as mod:
        #model = pickle.load(mod)

    # predict
    output = predict(text)

    # dictionary output for json
    #send_back = {'prediction': output}

    # give output to sender.
    return output