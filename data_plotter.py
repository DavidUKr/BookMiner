import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import datetime


nltk.download('vader_lexicon')
pd.set_option('display.max_rows', None)


def sentiment_chapter_plotting(dataframe):
    sia = SentimentIntensityAnalyzer()
    dfs = {book: df for book, df in dataframe.groupby('book')}
    
    fig = go.Figure()
    
    for book, df in dfs.items():
        list_nltk_sentiments = ['positive' if sia.polarity_scores(str(word))['compound'] > 0 else 'negative' if sia.polarity_scores(str(word))['compound'] < 0 else 'neutral' for word in df['word']]
        
        df = df.assign(nltk_sentiment = list_nltk_sentiments)
        
        df['sentiment_value'] = df['nltk_sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
        non_neutral_df = df[df['sentiment_value'] != 0]

        # Calculate the average sentiment per chapter
        avg_sentiment_per_chapter = non_neutral_df.groupby(['book','chapter'])['sentiment_value'].mean().reset_index()

        # Create the interactive line plot
        
    
        fig.add_trace(go.Scatter(x=avg_sentiment_per_chapter['chapter'],
                                y=avg_sentiment_per_chapter['sentiment_value'],
                                mode='lines+markers',
                                name=book))

    fig.update_layout(title='Average Sentiment per Chapter',
                    xaxis_title='Chapter',
                    yaxis_title='Average Sentiment',
                    hovermode='x')

    return fig

def treemap(categories,title,path,values):
    fig = px.treemap(categories, path=path, values=values, height=700,
                 title=title, color_discrete_sequence = px.colors.sequential.RdBu)
    fig.data[0].textinfo = 'label+text+value'
    return fig

def histogram(data,path,color,title,xaxis,yaxis):
    fig = px.histogram(data, x=path,color=color)
    fig.update_layout(
        title_text=title,
        xaxis_title_text=xaxis,
        yaxis_title_text=yaxis,
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig

def bar(categories,x,y,color,title,xlab,ylab):
    fig = px.bar(categories, x=x, y=y,
             color=color,
             height=400)
    fig.update_layout(
    title_text=title,
    xaxis_title_text=xlab,
    yaxis_title_text=ylab,
    bargap=0.2,
    bargroupgap=0.1
    )
    return fig

@st.cache_data
def plot_top_10_word_freq_total_bar(dataframe):
    return bar(dataframe,dataframe['word'][0:10],dataframe['count'][0:10],
    dataframe['word'][0:10],'Top 10 Frequent Words','Word','Count')
    
@st.cache_data
def plot_top_10_word_freq_per_book_hist(dataframe):
    return histogram(dataframe[:50], 'book', 'word', 'Word Frequency per book', 'Book', 'Word Counts')

@st.cache_data
def plot_tf_idf_scatterplot_3d(dataframe):
    dataframe=dataframe.sort_values(by='tf_idf', ascending=False)
    
    fig = px.scatter_3d(dataframe[:50], x='tf_idf', y='rank', z='word_appearances_in_book',
              color='book', size='tf_idf', size_max=18,
              symbol='word', opacity=0.7)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
    
@st.cache_data
def plot_tf_idf_scatterplot(dataframe):
    dataframe=dataframe.sort_values(by='tf_idf', ascending=False)
    
    fig = px.scatter(dataframe[:50], x='tf_idf', y='word_appearances_in_book', color='book', symbol="word",
                 size='tf_idf', title='TF-IDF Scatter Plot')
    return fig
    