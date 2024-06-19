import streamlit as st
import data_processor as dp
import data_plotter as dpl
import information_retrieval as ir
import pandas as pd
import requests
import topic_modeller as tm

#DATA processing script
@st.experimental_fragment
def display_word_freq(words_df, books_df):
    books_df=dp.sort_by_count(books_df)
    
    with st.container() as wf_container:
        viz_freq=st.radio(
            "What do you want to see?",
            ["Total", "Per Book", "Dataframe"],
            captions = ["Top 10 of all books", "Top 50 word freq per book", "Plain dataframe"],
            key="display_wf_radio",
            horizontal=True,
        )
        if viz_freq == "Total":
            st.write(dpl.plot_top_10_word_freq_total_bar(books_df))
        elif viz_freq == "Per Book":
            st.write(dpl.plot_top_10_word_freq_per_book_hist(words_df))
        else:
            st.write(words_df)
            
@st.experimental_fragment
def display_tf_idf(books_df):
    with st.container() as tf_idf_container:
        viz_freq=st.radio(
            "What seems interesting?",
            ["Scatter Plot","3D Scatter Plot", "Dataframe"],
            captions = ["Top 50 by tf_id", "Just Cooler:fire:", "Plain dataframe"],
            key="display_tfidf_radio",
            horizontal=True,
        )
        if viz_freq == "Scatter Plot":
            st.plotly_chart(dpl.plot_tf_idf_scatterplot(books_df))
        elif viz_freq == "3D Scatter Plot":
            st.plotly_chart(dpl.plot_tf_idf_scatterplot_3d(books_df))
        else:
            st.write(books_df)

def process_data(books_df):
    
    st.write("Exploding words...")
    books_df=dp.explode_to_words(books_df)
    books_df=books_df[['word', 'book']] #we only use word-book
    st.write(books_df)
    
    st.write("Filtering stopwords...")
    books_df=dp.filter_stopwords(books_df)
    st.write(books_df)
    
    st.write("Sorting by term frequency...")
    st.session_state.word_df=dp.sort_by_count_book(books_df)
    display_word_freq(st.session_state.word_df, books_df)

    st.write("Calculating tf/idf and rank...")
    st.session_state.books_df=dp.add_tf_idf(books_df)
    display_tf_idf(st.session_state.books_df)

def sentiment_analysis(books_df):
    st.write("Sentiment by chapter analysis...")
    
    books_df=dp.add_chapters(books_df)
    books_df=dp.explode_to_words(books_df)
    
    st.write(dpl.sentiment_chapter_plotting(books_df))
    
@st.experimental_fragment
def see_rec():
    top_idf=st.session_state.books_df.sort_values(by='tf_idf', ascending=False)
        
    query=""
    for word in top_idf['word'][:6]:
        query= query+" "+word
    
    with st.container() as retreival_container:
        retrieval_method=st.radio(
                "What method should I use?",
                ["Cosine Similarity", "BM25", "Latent Semantic Indexing"],
                key="retr_search_radio",
                horizontal=True,
            )
        
        indices, scores= ir.recommend_documents(query, retreival_method=retrieval_method)
        
        for idx, score in zip(indices, scores):
            with st.container(border=True):
                st.write(f"Document index: {idx}, BM25 Score: {score}\nDocument: {ir.texts[idx][:1000]}\n") 
#STREAMLIT


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

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Book Miner :books: :hammer_and_wrench:")

#declaring session state
if 'url_list' not in st.session_state:
    st.session_state["url_list"] = []

if 'books_df' not in st.session_state:
    st.session_state["books_df"] = pd.DataFrame()
    
if 'word_df' not in st.session_state:
    st.session_state["word_df"] = pd.DataFrame()


#DATA input
input_url=st.text_input(label="URL Source",placeholder="Paste URL to .txt format of book here"
                        , help = "Use urls from www.gutenberg.org")
submit_button=st.button(label="Submit :white_check_mark:")

with st.container() as submit_container:
    col_list, col_df = st.columns(2)
    
    if submit_button and input_url:
        st.session_state.url_list.append(input_url)

    elif submit_button:
        st.write("No URL. Please paste in some URL")
        

    # DATA cleaning and manipulation
    with col_list:
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

    with col_df:
        if not books_df.empty:
            st.write("Resulting dataframe:")
            st.write(books_df)

#DATA mining
with st.container() as process_container:
    if st.button(label="Process :hammer:") and not books_df.empty:
        process_data(books_df)

with st.container(border=True) as sentiment_container:
    if st.button(label="Sentiment analysis :hammer:") and not books_df.empty:
        sentiment_analysis(books_df)
        
with st.container(border=True) as article_recomandations_container:
    st.write("Recomended articles to read based on books topic")
    
    if st.button(label="See recomendations") and not st.session_state.books_df.empty:
        see_rec()

with st.container(border=True) as topic_modeling_container:
    st.write("Topic modeling on the submitted books")
    
    if st.button(label="Model Topics") and len(st.session_state.url_list)>0:
        for url in st.session_state.url_list:
            book=requests.get(url).text
            proc_book=tm.preprocess(book)
            tm.apply_LDA(proc_doc=proc_book)