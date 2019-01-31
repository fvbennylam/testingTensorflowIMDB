# source from https://www.tensorflow.org/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras

import numpy as np

# print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
# each label has an integer value of either 0 or 1
# 0 is a negative review
# 1 is a positive review

# print("Training entries: {}, labels:{}".format(len(train_data), len(train_labels)))

# each integers here represent a specific word in a dictionary
# print(train_data[0])

# this shows the number of words in the first and second reviews
# print(len(train_data[0]), len(train_data[1]))

# mapping words to an integer index
word_index = imdb.get_word_index()

# first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# this will now display the first review
# print(decode_review(train_data[0]))

# pad the arrays of integers to be the same length
# create an integer tensor of shape max_length * num_reviews
# use pad_sequences function to standardize the movie review lengths

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value = word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen =256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"],
padding = 'post', maxlen = 256)

# now all the reviews have the same lengths
# print(len(train_data[0]), len(train_data[1]))

# all the extra entries are filled with 0s
# print(train_data[0])

# building model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
# first layer takes the integer-encoded vocabulary and finds the embedding vector for each word-index
# these vectors are learned as model trains
model.add(keras.layers.Embedding(vocab_size, 16))
# this layer returns a fixed-length output vector for each example by averaging over the sequence dimension
# allows the model to handle input of variable length
model.add(keras.layers.GlobalAveragePooling1D())
# fixed-length output vector is piped through a dense layer with 16 hidden units
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# this layer is connected with a single output node which uses the sigmoid activation system
# to classify a float between 0 and 1, also representing a confidence level
model.add(keras.layers.Dense(1, activation= tf.nn.sigmoid))

# note: the number of outputs in a hidden layer represents the dimension of the representational space for the layer
#       more hidden units/layers (higher-dimensional representation space) allows the model to learn more complex representations
#       however it makes the network more computationally expensive and may lead to learning unwanted patterns (overfitting)

# print(model.summary())


model.compile(optimizer ='adam',
              loss = 'binary_crossentropy',
              metrics= ['accuracy'])


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose =1)

results = model.evaluate(test_data, test_labels)

print(results)


