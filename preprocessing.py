import numpy as np
import pandas as pd
from gensim import models
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn import preprocessing
import string

def convert_year(year):
    '''
    convert a year to year bin (e.g 2019 to 2010s)
    Return string represent year bin
    '''
    if year >2019:
        return '2020s'
    elif (year<2020) & (year> 2009):
        return '2010s'
    elif (year<2010) & (year> 1999):
        return '2000s'
    elif (year<2000) & (year> 1989):
        return '1990s'
    elif (year<1990) & (year> 1979):
        return '1980s'
    elif (year<1980) & (year> 1969):
        return '1970s'
    elif (year<1970) & (year> 1959):
        return '1960s'
    elif (year<1960) & (year> 1949):
        return '1950s'
    else:
        return None
    
def combine_sentences(token_series):
    '''
    combine a list of tokens to a list of sentence
    '''
    sentence = ' '.join(token_series)
    return sentence


def tokenize_lyric(text_series, stopword = False):
    '''
    Cleans, tokenizes + stems Pandas series of strings.
    
    Returns pandas series of lists of tokens
    '''
    # Clean text with regex
    clean = text_series.str.lower()
    #print(clean)
    #remove punctuations
    translate_table = dict((ord(char), None) for char in string.punctuation)   
    clean = clean.apply(lambda text: text.translate(translate_table).strip())
    # Anonymous tokenizer + stemmer functions
    if stopword:
        stop = nltk.corpus.stopwords.words('english')
        tokenize = lambda text: [i for i in nltk.word_tokenize(text) if i not in stop]
    else:
        tokenize = lambda text: [i for i in nltk.word_tokenize(text)]
    stemmer = lambda tokens: [SnowballStemmer('english').stem(token) for token in tokens]

    # Tokenize and stem clean text
    tokens = clean.apply(tokenize)
    stemmed_tokens = tokens.apply(stemmer)
    
    return stemmed_tokens