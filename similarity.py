import pandas as pd
import numpy as np
import heapq
from sklearn.feature_extraction.text import CountVectorizer
import numpy.linalg as la
from sklearn.utils.extmath import randomized_svd
from sklearn import preprocessing
from embedding import *
from preprocessing import * 
from scipy import stats



def Mens_Vect(word_ls,vocab_list_lyrics): #find common words in MENS and embedding tokens
    same_vocab = []
    for word in word_ls:
        if word in vocab_list_lyrics:
            same_vocab.append(word)
            
    return same_vocab

def Pair_Word(word,ppmi,svd_ppmi,vocab_list_lyrics,sim_word_pair): #find same word pairs in two embedding tokens and MENS word pairs
    love_ind = vocab_list_lyrics.index(word)
    ppmi_closest = closest(ppmi,love_ind, vocab_list_lyrics) 
    ppmi_dict = Into_Dict(ppmi_closest,word)
    
    svd_closet = closest(svd_ppmi,love_ind, vocab_list_lyrics)
    svd_dict = Into_Dict(svd_closet,word)
    
    #find same pairs in sim_word_pair
    same_word_dict_ppmi = Same_pair(ppmi_dict,sim_word_pair,PPMI=True)
    same_word_dict_svd = Same_pair(svd_dict,sim_word_pair,False)
    
    return same_word_dict_ppmi,same_word_dict_svd

def Same_pair(ppmi_dict,same_vocab,PPMI=False):# find the matched pairs PPMI / SVD similarity and MENS similarity 
    same_word_dict = []
    for ppmi in ppmi_dict:
        word1 = ppmi['word1']
        word2 = ppmi['word2']
        simi_ppmi = ppmi['simi']
        for same in same_vocab:
            word1_same = same['word1']
            #print(same)
            word2_same = same['word2']
            same_simi = same['simi']
            if ((word1 == word1_same) & (word2 == word2_same))| ((word1 == word2_same) & (word2 == word1_same)):
                if PPMI == True:
                    same_word =  {'word1':word1,'word2':word2,'PPMI cos':simi_ppmi,'MENS_simi':same_simi}
                else:
                    same_word =  {'word1':word1,'word2':word2,'SVD cos':simi_ppmi,'MENS_simi':same_simi}
                same_word_dict.append(same_word)
    return same_word_dict
    
def Into_Dict(ppmi_closest,word): #convert the word to dict
    ppmi_dict = []
    for ppmi in ppmi_closest:
        new_dict = {}
        simi = ppmi[0]
        word2 = ppmi[1]
        new_dict['word1'] = word
        new_dict['word2'] = word2
        new_dict['simi'] = simi
        ppmi_dict.append(new_dict)
    return ppmi_dict

def tokenize_mens(text):
    '''
    Cleans, tokenizes + stems Pandas series of Mens.
    
    Returns pandas series of lists of tokens
    '''
    tokens = [i for i in nltk.word_tokenize(text)]
    stemmed_tokens = [SnowballStemmer('english').stem(token) for token in tokens]
    return stemmed_tokens


def Simi_Pairs(total_match_pair): #calculate the simi pairs for svd/ppmi
    sum_PPMI = 0
    ppmi_mens_sim = 0
    ppmi_ls = []
    
    sum_SVD = 0
    svd_mens_sim = 0
    svd_ls = []
    
    for pair in total_match_pair:
        new_dict_ppmi = {}
        new_dict_svd = {}
        if 'PPMI cos' in pair:
            PPMI_simi = pair['PPMI cos']
            mens_simi = pair['MENS_simi']
            sum_PPMI += PPMI_simi
            ppmi_mens_sim += mens_simi
            
            new_dict_ppmi['PPMI_simi'] =PPMI_simi
            new_dict_ppmi['PPMI_MENS_simi'] = mens_simi
            ppmi_ls.append(new_dict_ppmi)
            
        elif 'SVD cos' in pair:
            SVD_simi = pair['SVD cos']
            mens_simi = pair['MENS_simi']
            sum_SVD += SVD_simi
            svd_mens_sim += mens_simi
            
            new_dict_svd['SVD_simi'] = SVD_simi
            new_dict_svd['SVD_MENS_simi'] = mens_simi
            svd_ls.append(new_dict_svd)
    return sum_PPMI,ppmi_mens_sim,sum_SVD,svd_mens_sim,ppmi_ls,svd_ls