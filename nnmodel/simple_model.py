
# coding: utf-8

# In[5]:





# In[6]:


from utils import *
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


# In[7]:


# %pdb


# In[8]:


glove = '../embeddings/glove_s300.txt'
log = '../embeddings/log_glove.txt'
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove, log)


# In[5]:


word = "constitucional"
index = 28989
print("the index of", word, "in the vocabulary is", words_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_words[index])


# In[11]:


# load X and Y
max_len = 100
data_files = get_ipython().getoutput('ls ../data')
X, Y = getXY(data_files)
Y_ohe, Y_idx, klass_to_idx, idx_to_klass = convert_to_one_hot(Y)
X_indices = sentences_to_indices(X, words_to_index, max_len)


# In[12]:


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
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
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


# In[13]:


embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# In[14]:


def create_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(17)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(input=sentence_indices, output=X)
    
    return model


# In[15]:


model = create_model((max_len,), word_to_vec_map, words_to_index)
model.summary()


# In[16]:


X_train, X_test, Y_train, Y_test = splitXY(X_indices, Y_ohe)


# In[17]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[18]:


model.fit(X_train, Y_train, epochs=50, batch_size=16, shuffle=True)


# In[19]:


model.evaluate(X_test, Y_test)


# In[20]:


model.evaluate(X_train, Y_train)


# In[21]:


model.evaluate(X_indices, Y_ohe)

