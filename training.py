import os
import json
import numpy as np 
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
from .utils import get_line_vector
from .utils import cosine_similarity, isc_similarity, correlation_similarity


# We have used the following LLM and model architecture for training
DEFAULT_bertModel = 'bert-base-multilingual-cased'
DEFAULT_tfModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, 768)),
    tf.keras.layers.Dense(768, activation='relu'),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
DEFAULT_simModel = isc_similarity

def trainModel(dataPath, bertModel=DEFAULT_bertModel, tfModel=DEFAULT_tfModel, simModel=DEFAULT_simModel, epochs=10):
    # downloading LLM model
    LLM_model = AutoModel.from_pretrained(bertModel, output_hidden_states=True, from_tf=False)
    LLM_tokenizer = AutoTokenizer.from_pretrained(bertModel, output_hidden_states=True, from_tf=False)

    # the dataset obtained from dataPath should be a .json file containing an array of dimensions (n x 2 x 2) - consisting of n pairs of sentences and each sentence is in an array of size 2, where 1st element is the sentence and 2nd element is the topic of the sentence
    file = open(dataPath)
    dataset = json.load(file)
    file.close()

    # pre-processing & generating ground truth
    xSet = np.array([[get_line_vector(s[0][0], LLM_tokenizer, LLM_model, s[0][1]), get_line_vector(s[1][0], LLM_tokenizer, LLM_model, s[1][1])] for s in dataset])
    ySet = np.array([simModel(s[0].reshape(1, -1), s[1].reshape(1, -1))[0][0] for s in xSet])

    # training process
    tfModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    tfModel.fit(xSet, ySet, epochs=epochs)
    return tfModel

def saveModel(tfModel, modelName):
    jsonModel = tfModel.to_json()
    with open(os.path.join("models", modelName, "model.json"), "w") as file:
        file.write(jsonModel)
    tfModel.save_weights(os.path.join("models", modelName, "weights.h5"))
    return
