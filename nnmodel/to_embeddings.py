# to_embeddings.py

from utils import *
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

glove = '../embeddings/glove_s300.txt'
log = '../embeddings/log_glove.txt'
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove, log)

# load X and Y
max_len = 100
data_files = ['JURIS-Enunciados1.xls', 'JURIS-Enunciados2.xls', 'JURIS-Enunciados3.xls']
X, Y = getXY(data_files)
# Y_ohe, Y_idx, klass_to_idx, idx_to_klass = convert_to_one_hot(Y)
X_indices = sentences_to_indices(X, words_to_index, max_len)

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["escape"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        try:
            emb_matrix[index, :] = word_to_vec_map[word]
        except:
            pass

    # Define Keras embedding layer with the correct output/input sizes. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def create_model(input_shape, word_to_vec_map, word_to_index):

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(input=sentence_indices, output=embeddings)
    
    return model

model = create_model((max_len,), word_to_vec_map, words_to_index)

X_embeddings = model.predict(X_indices)

np.save("data_embeddings", X_embeddings)