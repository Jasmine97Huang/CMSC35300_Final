import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import numpy.linalg as la
from sklearn.utils.extmath import randomized_svd

def get_co_mat(corpus, window_len =2):
    '''
    corpus: a list of sentences with words tokenized/lemmatized/lowered
    '''
    vect = CountVectorizer(stop_words = None, token_pattern = r'(?u)\b\w+\b')
    X = vect.fit_transform(corpus)
    # Create a co-occurrence matrix of unique words and initialize them to zero
    uniq_wrds = vect.get_feature_names()
    #print(uniq_wrds)
    n = len(uniq_wrds)
    co_mat = np.zeros((n,n))
    for sentence in corpus:
        update_co_mat(sentence, co_mat, uniq_wrds, window_len)
    return co_mat
    
def update_co_mat(sentence, co_mat, uniq_wrds, window_len =2):   
    # Get all the words in the sentence and store it in an array wrd_lst
    wrd_list = sentence.split(' ')
    
    # Consider each word as a focus word
    for focus_wrd_indx, focus_wrd in enumerate(wrd_list):
        if focus_wrd.isalpha():
            focus_wrd = focus_wrd.lower()
            # Get the indices of all the context words for the given focus word
            # focus word is counted as context word for itself
            for contxt_wrd_indx in range((max(0,focus_wrd_indx - window_len)),(min(len(wrd_list),focus_wrd_indx + window_len +1))):                        
                # If context words are in the unique words list
                if wrd_list[contxt_wrd_indx] in uniq_wrds:

                    # To identify the row number, get the index of the focus_wrd in the uniq_wrds list
                    co_mat_row_indx = uniq_wrds.index(focus_wrd)

                    # To identify the column number, get the index of the context words in the uniq_wrds list
                    co_mat_col_indx = uniq_wrds.index(wrd_list[contxt_wrd_indx])

                    # Update the respective columns of the corresponding focus word row
                    co_mat[co_mat_row_indx][co_mat_col_indx] += 1


def pmi(arr):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    #arr = df.as_matrix()
    
    # p(y|x) probability of each t1 overlap within the row
    row_totals = arr.sum(axis=1).astype(float) #counts of all occurrance for each word
    prob_cols_given_row = (arr.T / row_totals).T #prob one word (element of a col) given another word (row)
    #print(prob_cols_given_row)

    # p(y) probability of each t1 in the total set
    col_totals = arr.sum(axis=0).astype(float) 
    prob_of_cols = col_totals / sum(col_totals)
    #print('Sanity check, co-ocurrance mat is diagonal', np.equal(col_totals, row_totals))
    # PMI: log( p(y|x) / p(y) )
    # This is the same data, normalized
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001 #set 0 to a small num s.t. the log is negative.
    _pmi = np.log(ratio)
    _pmi[_pmi < 0] = 0

    return _pmi

def SVD_pmi(_pmi, n_components = 100, gamma = 0.5):
    '''
    Perform SVD on ppmi matrix to get the low-dimensional approximations of the PPMI embeddings
    gamma: the eigenvalue weighting parameter, setting gamme< 1 has been shown to dramatically improve embedding
    qualities
    n_componentsL desired dimensionality of output data. Must be strictly less than the number of features/unique tokens.
    
    '''
    U, S, Vt = randomized_svd(_pmi,n_components,random_state=42)
    if gamma == 0.0:
        SVD_emb = U
    elif gamma == 1.0:
        SVD_emb = S*U
    else:
        SVD_emb = np.power(S, gamma)*U
    #print(U)
    
    return SVD_emb