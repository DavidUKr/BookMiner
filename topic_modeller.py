import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
import pyLDAvis
import pyLDAvis.lda_model
import streamlit as st

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    
en_stop = set(nltk.corpus.stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')

warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

    return ' '.join(words)

def apply_LDA(proc_doc):
    vectorizer = CountVectorizer(max_df=0.75, min_df=2, stop_words='english', ngram_range=(1,3))
    term_document_matrix = vectorizer.fit_transform(proc_doc)

    n_topics = 4
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(term_document_matrix)
    
    # Prepare the LDA visualization data
    visualization_data = pyLDAvis.lda_model.prepare(lda, term_document_matrix, vectorizer)

    # Display the LDA visualization
    st.write(pyLDAvis.display(visualization_data))
