import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''
Main pipeline function for generation of the topic-informed attention guided sentence vector.
The get_line_vector() function works in two modes - 
    a. mean-pooling: if no key_word is mentioned then it will do mean pooling over the tokens
    b. topic-informed-pooling: you can specify the topic as string or a fixed length vector in the key_word attribute of the function
'''

def get_bert_embeddings(text, tokenizer, model):
    marked_text = '[CLS] ' + text + ' [SEP]'
    encode_text = tokenizer(marked_text, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
    model_output = model(**encode_text)
    output = np.array([x.detach().numpy() for x in model_output[2]])

    output = np.mean(output[-11:], axis=0)     # You can change the hidden layer pooling function here
    return output[0]

def get_line_vector(line: str, tokenizer, model, key_word=""):
    sentence_vec = get_bert_embeddings(line, tokenizer, model)
    ## 1 - MEAN POOLING
    # If no topic given, do normal mean pooling
    if key_word=="":
        sentence_vec = np.array(sentence_vec[1:-1])
        pooled = np.mean(sentence_vec, axis=0)
        return pooled
    
    ## 2 - TOPIC BASED ATTENTION
    # If there's a topic given then get the word vec for pooling
    if isinstance(key_word, str):
        vec = get_bert_embeddings(key_word, tokenizer, model)
        vec = np.array(vec[1:-1])
        word_vec = np.mean(vec, axis=0)
    else:
        word_vec = key_word

    # Word Weighted Pooling
    similarity = cosine_similarity(word_vec.reshape(1, -1), sentence_vec)[0]
    pooled = np.average(sentence_vec, axis=0, weights=similarity)
    return pooled