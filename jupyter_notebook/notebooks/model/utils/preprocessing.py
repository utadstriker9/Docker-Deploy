import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string
import pandas as pd

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('ascii') 
    text = text.strip()
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text) 
    
    # Replace unwanted characters
    text = re.sub(r'[\n\r\t]', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    text = re.sub(r"\b[a-zA-Z]\b", "", text)  
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    return text

def run_tokenizer(df,column_name=None):
    
    # Tokenizing
    def word_tokenize_wrapper(text):
        return word_tokenize(text)
        
    # Stopwords
    list_stopwords = stopwords.words("indonesian")
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", "klo", 
                           'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah'])
    list_stopwords = set(list_stopwords)
    
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)
        
    if isinstance(df, pd.DataFrame):  
        df[column_name+'_clean'] = df[column_name].fillna('').apply(clean_text)
        df[column_name+'_token'] = df[column_name+'_clean'].apply(word_tokenize_wrapper)
        df[column_name+'_token'] = df[column_name+'_token'].apply(stopwords_removal)
    
        term_dict = {}
    
        for document in df[column_name+'_token']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '
    
        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)
    
        def get_stemmed_term(document):
            return [term_dict[term] for term in document]
    
        df[column_name+'_token'] = df[column_name+'_token'].swifter.apply(get_stemmed_term)
        return df
    else:
        df = clean_text(df)
        df = word_tokenize_wrapper(df)
        df = stopwords_removal(df)
        df = [stemmed_wrapper(word) for word in df]
        return ' '.join(df)