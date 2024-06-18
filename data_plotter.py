import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import plotly.graph_objects as go

nltk.download('vader_lexicon')

def sentiment_chapter_plotting(dataframe):
    sia = SentimentIntensityAnalyzer()
    list_nltk_sentiments = ['positive' if sia.polarity_scores(str(word))['compound'] > 0 else 'negative' if sia.polarity_scores(str(word))['compound'] < 0 else 'neutral' for word in dataframe['word']]
    
    dataframe = dataframe.assign(nltk_sentiment = list_nltk_sentiments)
    
    dataframe['sentiment_value'] = dataframe['nltk_sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    non_neutral_df = dataframe[dataframe['sentiment_value'] != 0]

    # Calculate the average sentiment per chapter
    avg_sentiment_per_chapter = non_neutral_df.groupby('chapter')['sentiment_value'].mean().reset_index()

    # Create the interactive line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=avg_sentiment_per_chapter['chapter'],
                            y=avg_sentiment_per_chapter['sentiment_value'],
                            mode='lines+markers',
                            name='Average Sentiment'))

    fig.update_layout(title='Average Sentiment per Chapter',
                    xaxis_title='Chapter',
                    yaxis_title='Average Sentiment',
                    hovermode='x')

    return fig