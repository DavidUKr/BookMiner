from sklearn.datasets import fetch_20newsgroups
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from gensim import corpora, models, similarities

nltk.download('punkt')

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')


# Access the data
texts = newsgroups.data  # the actual newsgroup postings
target = newsgroups.target  # the category labels
target_names = newsgroups.target_names  # the names of the categories

from sklearn.feature_extraction.text import TfidfVectorizer

# Setup the vectorizer with necessary preprocessing settings
vectorizer = TfidfVectorizer(
    lowercase=True,        # Convert all characters to lowercase before tokenizing
    stop_words='english',  # Remove stopwords
    max_df=0.95,           # Terms that appear in more than 95% of the documents are ignored
    min_df=2,              # Terms that appear in less than 2 documents are ignored
    max_features=10000     # Only consider the top 10,000 features ordered by term frequency across the corpus
)

# Apply vectorization to the text data
tfidf_matrix = vectorizer.fit_transform(texts)

tokenized_corpus = [word_tokenize(doc.lower()) for doc in texts]

# Create BM25 object
bm25 = BM25Okapi(tokenized_corpus)

dictionary = corpora.Dictionary(tokenized_corpus)
corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

# Build the LSI model
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=300)

# Build the index
index = similarities.MatrixSimilarity(lsi[corpus])
    
def retrieve_cos_similarity(query, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, top_k=5):
    # Transform the query to the same vector space as the documents
    query_vec = vectorizer.transform([query])

    # Compute the cosine similarity between query vector and all document vectors
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get the top k documents with the highest similarity scores
    top_indices = np.argsort(-similarities)[:top_k]  # argsort returns indices of sorted array

    return top_indices, similarities[top_indices]

def retrieve_bm25(query, top_k=5):
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices, scores[top_indices]

def retrieve_lsi(query, top_k=5):
    query_bow = dictionary.doc2bow(word_tokenize(query.lower()))
    query_lsi = lsi[query_bow]
    similarities = index[query_lsi]
    top_indices = np.argsort(-similarities)[:top_k]
    return top_indices, similarities[top_indices]

def recommend_documents(query, retreival_method, top_k=5):
    
    if retreival_method == "Cosine Similarity":
        return retrieve_cos_similarity(query, top_k=top_k)
    elif retreival_method == "BM25":
        return retrieve_bm25(query, top_k=top_k)
    else:
        return retrieve_lsi(query, top_k=top_k)
