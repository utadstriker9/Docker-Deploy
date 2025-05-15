# Sentiment Analysis 

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd 
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import preprocessing as pr

def sentiment_analysis(df,column_name='text'):
    sa_pretrained= "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(sa_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(sa_pretrained)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

    if isinstance(df, pd.DataFrame): 
        df[column_name] = df[column_name]
    else:
        df = pd.DataFrame({column_name: [df]})
   
    df[column_name+'_clean'] = df[column_name].fillna('').apply(pr.clean_text)
    for index, row in df.iterrows():
        text = row[column_name+'_clean']
        result = sentiment_analysis(text)[0]
        df.loc[index, 'label_sentiment'] = label_index[result['label']]
    return df

# Klasifikasi Review

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import preprocessing as pr
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

sysdir = os.path.dirname(os.path.abspath(__file__))

file_tfidf_vectorizer = os.path.join(sysdir, "utils", "tfidf_vectorizer.pkl")
file_model_klasifikasi = os.path.join(sysdir, "utils", "model_klasifikasi.pkl")
file_label_encoder = os.path.join(sysdir, "utils", "label_encoder.pkl")

def review_classification(df,column_name='text'):
    if isinstance(df, pd.DataFrame): 
        df[column_name] = df[column_name]
    else:
        df = pd.DataFrame({column_name: [df]})
    
    X_clean = pr.run_tokenizer(df,column_name)
    X_clean = [" ".join(tokens) for tokens in X_clean[column_name+'_token']]
        
    tfidf_vectorizer = joblib.load(file_tfidf_vectorizer)
    X_clean = tfidf_vectorizer.transform(X_clean)
    model = joblib.load(file_model_klasifikasi)
    y_pred = model.predict_proba(X_clean)
    y_pred.tolist()

    label_encoder = joblib.load(file_label_encoder)

    class_labels = label_encoder.inverse_transform(
            np.arange(len(label_encoder.classes_))
        )
    predicted_topics = [class_labels[np.argmax(probs)] for probs in y_pred]
    df["topics"] = predicted_topics
    return df