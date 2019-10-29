import nltk
import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import Error
pd.options.display.float_format = '{:,}'.format

DIR = '../data'

df_reviews = pd.read_csv(os.path.join(DIR, 'df_reviews_short.csv'))

print(df_reviews.shape)

df_reviews = df_reviews.head(1000)

# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

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


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", \
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", \
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", \
                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", \
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

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

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", \
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", \
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", \
                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", \
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", \
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", \
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  \
                    "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", \
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", \
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",\
                    "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not",\
                     "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",\
                     "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", \
                     "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", \
                     "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", \
                     "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",\
                     "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", \
                     "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",\
                     "they'd": "they would", "they'd've": "they would have", "they'll": "they will", \
                     "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", \
                     "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", \
                     "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", \
                     "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", \
                     "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", \
                     "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", \
                     "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", \
                     "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",\
                     "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", \
                     "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", \
                     "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

df_reviews["review_clean"] = 1

for i in range (len(df_reviews)):
    df_reviews["review_clean"][i] = clean_text(df_reviews["review_text"][i])
    if (i % 10 == 0 or i <= 30):
        print(i, end=',\n')

export_csv = df_reviews.to_csv (r'../data/df_reviews1000.csv', \
 	index = None, header=True)