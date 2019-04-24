"""
Lee Alessandrini

Text Mining
"""

import sys
import argparse
import pickle
import gensim
import numpy as np
# Scikit-learn
from sklearn.model_selection import train_test_split
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Embedding, LSTM

from clean_tweets import clean_tweets


# --- Setup Constants ---
# DATASET values
TRAIN_SIZE = 0.8
# WORD2VEC values
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10
# KERAS values
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024
# SENTIMENT values
POSITIVE = "+"
NEGATIVE = "-"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
# EXPORT values
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"


def split_data(data):
    """
        This method will split the data set randomly.

        Args:
            data (pandas.DataFrame): data table

        Returns:
            randomized training and test sets
    """
    training_set, test_set = train_test_split(
        data, test_size=1 - TRAIN_SIZE, random_state=42)

    return training_set, test_set


def build_vocab_vector(training_set):
    """

    """
    # Build vocab vector using training set
    documents = [_text.split() for _text in training_set.text]

    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)
    w2v_model.build_vocab(documents)
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

    return w2v_model


def build_tokenizer(training_set):
    """

    """

    # Tokenize Text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_set.text)
    vocab_size = len(tokenizer.word_index) + 1

    return tokenizer, vocab_size


def tokenize_text(tokenizer, data_set):
    """

    """

    # Get x and y vectors
    x_vector = pad_sequences(tokenizer.texts_to_sequences(data_set.text),
                             maxlen=SEQUENCE_LENGTH)
    y_vector = data_set['target'].values.reshape(-1, 1)

    return x_vector, y_vector


def build_model(w2v_model, tokenizer, vocab_size, x_train, y_train):
    """
        This method will build the model given a training set.
    """

    # Build embedding layer
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    embedding_layer = Embedding(
        vocab_size, W2V_SIZE, weights=[embedding_matrix],
        input_length=SEQUENCE_LENGTH, trainable=False)
    # Build model
    model = models.Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
    # Train
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,
        callbacks=callbacks)

    return model


def score_model(model, x_test, y_test):
    """

    """
    # Score model
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("ACCURACY:", score[1])
    print("LOSS:", score[0])

    return


def save_model(model, w2v_model, tokenizer):
    """

    """
    # Save model
    model.save(KERAS_MODEL)
    w2v_model.save(WORD2VEC_MODEL)
    pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
    #pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)


def main(filepath, build, load, score, predict):
    # Read in data
    data = clean_tweets(filepath)
    # Filter to 1000 for testing
    #data = data.loc[:1000]
    # If scoring model, split data set
    if score:
        training_set, test_set = split_data(data)
    else:
        training_set = data
    # Build vocab vector model
    w2v_model = build_vocab_vector(training_set)
    # Build tokenizer and get vocab size
    tokenizer, vocab_size = build_tokenizer(training_set)
    # Tokenize text
    x_train, y_train = tokenize_text(tokenizer, training_set)
    # Build model
    if build:
        model = build_model(w2v_model, tokenizer, vocab_size,
                            x_train, y_train)
        # Save model
        save_model(model, w2v_model, tokenizer)
    elif load:
        # Add model loading
        model = models.load_model(load)
    else:
        print('Select build or load model.')
        sys.exit()
    # Score model
    if score:
        x_test, y_test = tokenize_text(tokenizer, test_set)
        score_model(model, x_test, y_test)
    elif predict:
        # Add prediction logic
        # Predict
        # model.predict([x_test])
        pass
    else:
        print('Select score or predict.')
        sys.exit()


if __name__ == '__main__':
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Shoutout Jack Dorsey!")
    parser.add_argument('-f', '--file-path', dest='filepath', action='store',
                        default='Training.txt',
                        help='Give file path for training data.')
    parser.add_argument('-bm', '--build-model', dest='build', action='store_true',
                        help='Build model and save as h5 file.')
    parser.add_argument('-lm', '--load-model', dest='load', action='store',
                        help='Load h5 keras model.')
    parser.add_argument('-s', '--score', dest='score', action='store_true',
                        help='Score the model')
    parser.add_argument('-p', '--predict', dest='predict', action='store',
                        help='Predict given path with the model')
    # Cast args to dict
    args = vars(parser.parse_args(sys.argv[1:]))

    main(**args)
