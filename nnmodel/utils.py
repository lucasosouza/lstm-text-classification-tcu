import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split

def splitXY(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def read_glove_vecs(glove_file, log_file):

    with open(log_file, 'w') as log:
        with open(glove_file, 'r') as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                words_line = line.strip().split()
                weights = words_line[-300:]
                curr_word  = ''.join(words_line[:-300])
                try:
                    word_to_vec_map[curr_word] = np.array(weights, dtype=np.float64)
                    words.add(curr_word)
                except:
                    log.write(line)
            i = 1
            words_to_index = {}
            index_to_words = {}
            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1

    return words_to_index, index_to_words, word_to_vec_map

# def read_glove_vecs(glove_file, log_file):

#     with open(log_file, 'w') as log:
#         with open(glove_file, 'r') as f:
#             words = set()
#             word_to_vec_map = {}
#             for line in f:
#                 words_line = line.strip().split()
#                 curr_word = words_line[0]
#                 try:
#                     word_to_vec_map[curr_word] = np.array(words_line[1:], dtype=np.float64)
#                     words.add(curr_word)
#                 except:
#                     log.write(line)
#             i = 1
#             words_to_index = {}
#             index_to_words = {}
#             for w in sorted(words):
#                 words_to_index[w] = i
#                 index_to_words[i] = w
#                 i = i + 1

#     return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getXY(data_files):
    X = pd.Series()
    Y = pd.Series()
    for file in data_files:
        df = pd.read_excel('../data/' + file)
        X = pd.concat([X, df[' Enunciado ']], ignore_index=True)
        Y = pd.concat([Y, df[' Área ']], ignore_index=True)

    return X, Y


def convert_to_one_hot(Y):
    nclasses = len(Y.value_counts())
    klass_to_idx = {klass: idx for idx, klass in enumerate(Y.value_counts().index)}
    idx_to_klass = {idx: klass for idx, klass in enumerate(Y.value_counts().index)}
    Y_idx = Y.apply(lambda w: klass_to_idx[w])
    Y_ohe = np.eye(nclasses)[Y_idx]
    return Y_ohe, Y_idx, klass_to_idx, idx_to_klass

def sentences_to_indices(X, word_to_index, max_len, log_file='log_wordindex.txt', debug_file='debug_wordindex.txt'):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples

    # set to remove punctuation
    exclude = set(string.punctuation)
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    with open(debug_file, 'w') as f2:
        with open(log_file, 'w') as f:
            for i in range(m):                               # loop over training examples
                
                # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
                sentence = ''.join(ch for ch in X[i] if ch not in exclude)
                sentence_words = sentence.lower().split()
                
                # Initialize j to 0
                j = 0
                
                # Loop over the words of sentence_words
                for w in sentence_words:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    try:
                        X_indices[i, j] = word_to_index[w]
                        f2.write(w+'\n')
                    except:
                        f.write(w+'\n')
                    # Increment j to j + 1
                    j = j + 1
                    
    
    return X_indices


def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


