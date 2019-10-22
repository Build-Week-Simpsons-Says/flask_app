import numpy as np
import pandas as pd
import re
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# load dataframe
df0 = pd.read_csv('simpsons_dataset.csv')

# drops cols with null values and resets index
df = df0.dropna().copy()

# creates natural language processor
nlp = spacy.load('en_core_web_lg')

# creates set of stopwords
STOPWORDS = nlp.Defaults.stop_words.union({' ', '', 've'})

# creates tokenizer
tok = Tokenizer(nlp.vocab)


def lemmatize(doc):
    '''
    function to remove stop words and return list of lemmas from an input doc
    '''
    text = re.sub(r'[^a-zA-Z ]', '', doc)
    text = text.lower()
    tokens = tok(text)
    list_of_tokens = [t.lemma_ for t in tokens if ((str(t) not in STOPWORDS)
                                                   and (t.is_punct is False))]
    return(list_of_tokens)


# creates lemma feature
df['lemmas'] = df['spoken_words'].apply(lemmatize)

# gets number of lemmas in each quote
df['lemma_count'] = df['lemmas'].apply(len)

# filters out quotes with less than 3 lemams
df_filtered = df[df['lemma_count'] >= 3].copy()

# drops the count column as it will not be used again
df_filtered.drop(columns=['lemma_count'], inplace=True)

# resets index in order to find quotes with index value later
df_filtered.reset_index(drop=True, inplace=True)


# defines function to turn lemmas into str for use in vectorizer
def lem_str(lem):
    new_str = ''
    for word in lem:
        new_str += word + ' '
    return(new_str)


# creates vectorizer
tfidf = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.80)

# creates sparse matrix of lemmas
sparse = tfidf.fit_transform(df_filtered['lemmas'].apply(lem_str))

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
    '''

    # creates vector of input text
    quote_vec = tfidf.transform([input_text])

    # gets the index for the most similar quote to the input text
    similar_index = nn_model.kneighbors(quote_vec.todense())[1][0][0]

    # gets the quote from the dataframe
    output_quote = df_filtered['spoken_words'][similar_index]

    # returns quote
    return(output_quote)
