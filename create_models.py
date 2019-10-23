import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump

# load dataframe
df = pd.read_csv('filtered_dataset.csv')

# creates vectorizer
tfidf = TfidfVectorizer(stop_words='english',
                        lowercase=True,
                        min_df=0.003)

# creates sparse matrix of lemmas
sparse = tfidf.fit_transform(df['spoken_words'])

# creates dense matrix from sparse matrix
dense = sparse.todense()

# creates dataframe from dense matrix
dense_df = pd.DataFrame(dense, columns=tfidf.get_feature_names())

# creates NearestNeighbors model and fits it to dense dataframe
nn_model = NearestNeighbors(n_neighbors=24,
                            leaf_size=358,  # arbitrary number
                            algorithm='kd_tree')
nn_model.fit(dense_df)

# pickles the models for use in app to avoid training more than once
dump(tfidf, 'tfidf.joblib')
dump(nn_model, 'nn_model.joblib')
