import requests
import string as str
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def get_book_from_url(book_url):
    
    response = requests.get(book_url)
    book = response.text
    allowed_chars = str.ascii_letters + str.digits + str.whitespace
    book = ''.join(c for c in book if c in allowed_chars)
    
    return book

def get_df_from_book(book):
    book_lines = book.splitlines()

    book_df = pd.DataFrame({
        "line": book_lines,
        "line_number": list(range(len(book_lines)))
    })
    
    return book_df

def get_title(book):
    
    title_pattern = re.compile(r'Title\s*(.+)', re.IGNORECASE)
    match=title_pattern.search(book.splitlines()[10])

    if match:
        return match.group(1).strip()   
    else:       
        return "Title not found"

def explode_to_words(books_df):
    books_df['word'] = books_df['line'].str.split()
    books_df = books_df.explode('word')
    books_df = books_df.reset_index(drop=True)
    
    return books_df

def filter_stopwords(words_df):
    words_df = words_df[~words_df['word'].str.lower().isin(stopwords.words('english'))]
    words_df = words_df.dropna()
    return words_df

def sort_by_count(words_df):
    return words_df.groupby(['word', 'book']).size().sort_values(ascending=False).reset_index(name='count')

def add_tf_idf(book_words_df):
    count_df_1 = book_words_df.groupby(['word', 'book']).size().sort_values(ascending=False).reset_index(name='count') # How many appearances each word has in each book

    count_df_2 = book_words_df.groupby(['book']).size().sort_values(ascending=False).reset_index(name='count') # How many words each book has


    book_words = count_df_1.merge(count_df_2, on='book')

    book_words = book_words.rename(columns={'count_x': 'word_appearances_in_book', 'count_y': 'book_total_word_count'}) # Give more meaningful names

    book_words=book_words.assign(tf=book_words['word_appearances_in_book'] / book_words['book_total_word_count'])

    book_words['rank'] = book_words.groupby('book')['word_appearances_in_book'].rank(ascending=False, method='dense')

    import math
    N=4

    book_words = book_words.assign(idf=book_words.groupby(['word'])['book'].transform(lambda x:  math.log(N/len(x))))

    book_words=book_words.assign(tf_idf=book_words['idf']*book_words['tf'])
    
    return book_words

def line_is_chapter(dataframe):
    chapter_list = []
    curr_chapter = 0
    for index, row in dataframe.iterrows():
        if re.search("^chapter [\\divxlc]*$", row['line'], re.IGNORECASE):
            curr_chapter += 1
        chapter_list.append(curr_chapter)
    return chapter_list

def add_chapters(book_df):
    return book_df.assign(chapter = line_is_chapter(book_df))