import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import re
import nltk


sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

# Punctuation list
punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

##
## Same regex as imnported glove 
##

re_remove_brackets = re.compile(r'\{.*\}')
re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
re_transform_numbers = re.compile(r'\d', re.UNICODE)
re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
re_tree_dots = re.compile(u'…', re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                       (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                         (punctuations, punctuations), re.UNICODE)
re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
re_changehyphen = re.compile(u'–')
re_doublequotes_1 = re.compile(r'(\"\")')
re_doublequotes_2 = re.compile(r'(\'\')')
re_trim = re.compile(r' +', re.UNICODE)

def clean_text(text):
    """Apply all regex above to a given string."""
    text = text.lower()
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()

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
                curr_word = words_line[0]
                try:
                    word_to_vec_map[curr_word] = np.array(words_line[1:], dtype=np.float64)
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
    #exclude = set(string.punctuation)
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    with open(debug_file, 'w') as f2:
        with open(log_file, 'w') as f:
            for i in range(m):                               # loop over training examples
                
                # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
                #sentence = ''.join(ch for ch in X[i] if ch not in exclude)
                line = clean_text(X[i])                
                                
                words = []
                
                for sent in sent_tokenizer.tokenize(line):
                    
                    if sent.count(' ') >= 3 and sent[-1] in ['.', '!', '?', ';']:
                        if sent[0:2] == '- ':
                            sent = sent[2:]
                    elif sent[0] == ' ' or sent[0] == '-':
                        sent = sent[1:]
                    for w in sent.lower().split():
                        words.append(w)
                
                # Initialize j to 0
                j = 0
                
                # Loop over the words of sentence_words
                for w in words:
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


