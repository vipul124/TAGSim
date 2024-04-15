import os
import json
import numpy as np 
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
from pipeline import get_line_vector
from baselines import cosine_similarity, isc_similarity, correlation_similarity


# We have used the following LLM for embeddings
DEFAULT_bertModel = 'bert-base-multilingual-cased'
DEFAULT_modelPath = './models/TAGSim-base'

def testModel(dataPath='./dataset/testSet1_WIKI.json', bertModel=DEFAULT_bertModel, modelPath=DEFAULT_modelPath):
    # downloading LLM model
    LLM_model = AutoModel.from_pretrained(bertModel, output_hidden_states=True, from_tf=False)
    LLM_tokenizer = AutoTokenizer.from_pretrained(bertModel, output_hidden_states=True, from_tf=False)

    # extracting trained metric / TAGSim
    model = tf.keras.models.model_from_json(open(os.path.join(modelPath, "model.json")).read())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    model.load_weights(os.path.join(modelPath, "weights.h5"))

    # opening the dataset file
    file = open(dataPath)
    dataset = json.load(file)
    file.close()
    
    # converting into embeddings
    ss_in = np.array([[get_line_vector(s[0], LLM_tokenizer, LLM_model), get_line_vector(s[1], LLM_tokenizer, LLM_model)] for s in dataset['ss']])
    ds_in = np.array([[get_line_vector(s[0], LLM_tokenizer, LLM_model), get_line_vector(s[1], LLM_tokenizer, LLM_model)] for s in dataset['ds']])

    # get the results
    print("TAGSim:\t\t\t", "ss -", model.predict(ss_in).mean(), "\t\t;ds -", model.predict(ds_in).mean())
    print("Cosine Similarity:\t", "ss -", np.array([cosine_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ss_in]).mean(), "\t\t;ds -", np.array([cosine_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ds_in]).mean())
    print("ISC Similarity:\t\t", "ss -", np.array([isc_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ss_in]).mean(), "\t\t;ds -", np.array([isc_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ds_in]).mean())
    print("Correlation Similarity:\t", "ss -", np.array([correlation_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ss_in]).mean(), "\t;ds -", np.array([correlation_similarity(x[0].reshape(1, -1), x[1].reshape(1, -1))[0][0] for x in ds_in]).mean())