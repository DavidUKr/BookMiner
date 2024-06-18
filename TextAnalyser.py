import streamlit as st
import data_processor as dp
import data_plotter as dpl
import pandas as pd

#DATA processing script
def process_data(books_df):
    
    st.write("Exploding words...")
    books_df=dp.explode_to_words(books_df)
    books_df=books_df[['word', 'book']] #we only use word-book
    st.write(books_df)
    
    st.write("Filtering stopwords...")
    books_df=dp.filter_stopwords(books_df)
    st.write(books_df)
    
    st.write("Sorting by term frequency...")
    words_df=dp.sort_by_count(books_df)
    st.write(words_df)

    st.write("Calculating tf/idf and rank...")
    books_df=dp.add_tf_idf(books_df)
    st.write(books_df)

def sentiment_analysis(books_df):
    st.write("Sentiment by chapter analysis...")
    
    books_df=dp.add_chapters(books_df)
    books_df=dp.explode_to_words(books_df)
    
    st.write(dpl.sentiment_chapter_plotting(books_df))
    
#STREAMLIT

#setup
st.set_page_config(
    page_title="BookMiner",
    page_icon="static/books.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Made with dedication by *David Urs* and *Cristina Rodean*"
    }
)


st.title("Book Miner :books: :hammer_and_wrench:")

#declaring session state
if 'url_list' not in st.session_state:
    st.session_state["url_list"] = []

if 'book_df' not in st.session_state:
    st.session_state["book_df"] = pd.DataFrame()

#DATA input
input_url=st.text_input(label="URL Source",placeholder="Paste URL to .txt format of book here"
                        , help = "Use urls from www.gutenberg.org")
submit_button=st.button(label="Submit :white_check_mark:")

if submit_button and input_url:
    st.session_state.url_list.append(input_url)

elif submit_button:
    st.write("No URL. Please paste in some URL")
    

# DATA cleaning and manipulation
if len(st.session_state.url_list)>0:
    st.markdown("Submitted books:")

book_dfs=[]

for url in st.session_state.url_list:
    
    book=dp.get_book_from_url(url)
    book_df=dp.get_df_from_book(book)
    book_df=book_df.assign(book=dp.get_title(book))
    
    st.write(f"{dp.get_title(book)}\n URL:{url}")
    
    book_dfs.append(book_df)

books_df=pd.DataFrame()

if len(book_dfs)>0:    
    books_df=pd.concat(book_dfs)    

if not books_df.empty:
    st.write("Resulting dataframe:")
    st.write(books_df)

#DATA mining
if st.button(label="Process :hammer:") and not books_df.empty:
    process_data(books_df)
    
if st.button(label="Sentiment analysis :hammer:") and not books_df.empty:
    sentiment_analysis(books_df)