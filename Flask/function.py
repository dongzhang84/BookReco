from nltk.corpus import wordnet
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re
import gzip
import json
import os
import warnings

import matplotlib.pyplot as plt
from skimage import io

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

warnings.filterwarnings('ignore')

path_model  = '/Users/dong/Insight/Goodreads/models/'

dv = Doc2Vec.load(path_model+"doc2vec_model")
tf = pickle.load(open(path_model+"tfidf_model.pkl", "rb"))
svd = pickle.load(open(path_model+"svd_model.pkl", "rb"))
svd_feature_matrix = pickle.load(open(path_model+"lsa_embeddings.pkl", "rb"))
doctovec_feature_matrix = pickle.load(open(path_model+"doctovec_embeddings.pkl", "rb"))
hal = sia()

def query_similar_books(message, n):
    
    love_message, hate_message = get_message_sentiment(message)
    
    similar_books = get_ensemble_similarity_scores(love_message)
    dissimilar_books = get_dissimilarity_scores(hate_message)
    dissimilar_books = dissimilar_books.query('dissimilarity > .3')
    similar_books = similar_books.drop(dissimilar_books.index)
    
    return similar_books.head(n)


def get_message_sentiment(message):
    sentences = re.split('\.|\but',message)
    sentences = [x for x in sentences if x != ""]
    love_message = ""
    hate_message = ""
    for s in sentences:
        sentiment_scores = hal.polarity_scores(s)
        if sentiment_scores['neg'] > 0:
            hate_message = hate_message + s
        else:
            love_message = love_message + s
    return love_message, hate_message
      
      

def get_ensemble_similarity_scores(message):
    message = clean_text(message)
    bow_message_array = get_message_tfidf_embedding_vector(message)
    semantic_message_array = get_message_doctovec_embedding_vector(message)
    
    bow_similarity = get_similarity_scores(bow_message_array, svd_feature_matrix)
    semantic_similarity = get_similarity_scores(semantic_message_array, doctovec_feature_matrix)
    
    ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
    ensemble_similarity.columns = ["semantic_similarity", "bow_similarity"]
    ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"]                                                   + ensemble_similarity["bow_similarity"])/2
    ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
    return ensemble_similarity


def get_message_tfidf_embedding_vector(message):
    message_array = tf.transform([message]).toarray()
    message_array = svd.transform(message_array)
    message_array = message_array[:,0:25].reshape(1, -1)
    return message_array


def get_message_doctovec_embedding_vector(message):
    message_array = dv.infer_vector(doc_words=message.split(" "), epochs=200)
    message_array = message_array.reshape(1, -1)
    return message_array


def get_similarity_scores(message_array, embeddings):
    cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,Y=message_array,dense_output=True))
    cosine_sim_matrix.set_index(embeddings.index, inplace=True)
    cosine_sim_matrix.columns = ["cosine_similarity"]
    return cosine_sim_matrix


def get_dissimilarity_scores(message):
    message = clean_text(message)
    bow_message_array = get_message_tfidf_embedding_vector(message)
    semantic_message_array = get_message_doctovec_embedding_vector(message)
    
    dissimilarity = get_similarity_scores(bow_message_array, svd_feature_matrix)
    dissimilarity.columns = ["dissimilarity"]
    dissimilarity.sort_values(by="dissimilarity", ascending=False, inplace=True)
    return dissimilarity


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",                     "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",                     "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",                     "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",                     "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    text = replace_contractions(text)
    return(text)

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",                     "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",                     "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",                     "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",                     "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would",                     "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",                     "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",                      "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",                     "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",                     "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",                    "mightn't": "might not","mightn't've": "might not have", "must've": "must have",                     "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",                     "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",                     "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",                     "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",                     "there'd": "there would", "there'd've": "there would have", "there's": "there is",                     "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",                     "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",                     "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",                     "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",                     "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",                     "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",                     "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",                     "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",                     "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",                    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",                     "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)