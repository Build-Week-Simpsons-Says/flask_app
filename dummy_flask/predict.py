import numpy as np
import pandas as pd
from joblib import load

# load dataframe
df = pd.read_csv('filtered_dataset.csv')

# loads pickled TfidfVectorizer model
tfidf = load('tfidf.joblib')

# loads pickled nearest neighbor model
nn_model = load('nn_model.joblib')


def get_quote(input_text):
    '''
    function to find most similar quote to input text
    if input text is blank, a quote is chosen at random
    row of dataframe containing similar quote is returned in json format
    '''

    if input_text == '':

        # gets the index of a random quote in the DataFrame
        rand_index = np.random.randint(len(df))

        # gets the row containing the quote
        quote_row = df.iloc[rand_index]

        # converts output to json
        json_output = quote_row.to_json()

        return(json_output)

    else:
        # creates vector of input text
        quote_vec = tfidf.transform([input_text])

        # gets an object that contains the index values for similar quotes
        similar_quotes = nn_model.kneighbors(quote_vec.todense())[1][0]

        # generates a number to select out of the most similar quotes at random
        i = np.random.randint(len(similar_quotes))

        # gets the index value for the selected quote
        similar_index = similar_quotes[i]

        # gets the quote from the dataframe and returns it
        quote_row = df.iloc[similar_index]

        # converts output to json
        json_output = quote_row.to_json()

        return(json_output)
