import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# load dataframe
df = pd.read_csv('filtered_dataset.csv')

# creates vectorizer
tfidf = TfidfVectorizer(stop_words='english',
                        min_df=0.01,
                        max_df=0.80,
                        ngram_range=(1, 3))

# creates sparse matrix of lemmas
sparse = tfidf.fit_transform(df['spoken_words'])

# creates dense matrix from sparse matrix
dense = sparse.todense()

# creates dataframe from dense matrix
dense_df = pd.DataFrame(dense, columns=tfidf.get_feature_names())

# creates NearestNeighbors model and fits it to dense dataframe
nn_model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
nn_model.fit(dense_df)


def get_quote(input_text):
    '''
    function to find most similar quote to input text
    if input text is blank, a quote is chosen at random
    '''

    if input_text == '':

        # gets the index of a random quote in the DataFrame
        rand_index = np.random.randint(len(df))

        # gets the quote for that index and retunrs it
        rand_quote = df['spoken_words'][rand_index]
        return(rand_quote)

    else:
        # creates vector of input text
        quote_vec = tfidf.transform([input_text])

        # gets the index for the most similar quote to the input text
        similar_index = nn_model.kneighbors(quote_vec.todense())[1][0][0]

        # gets the quote from the dataframe and returns it
        output_quote = df['spoken_words'][similar_index]
        return(output_quote)
