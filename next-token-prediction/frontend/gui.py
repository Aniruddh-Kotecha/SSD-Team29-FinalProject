
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import os
import tokenize
from nltk.tokenize import word_tokenize
import re
import pickle
# import nltk
# nltk.download('punkt')

def nltk_tokenize_code(code):
    return word_tokenize(code)

with open('cleaned_code.txt', 'r', encoding='utf-8') as f:
    cleaned_code = f.read()
tokens = nltk_tokenize_code(cleaned_code)
vocab1 = sorted(set(tokens)) 
token_to_idx = {token: idx for idx, token in enumerate(vocab1)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

@st.cache_resource
def load_lstm_model(path):
    model = load_model(path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_data
def load_vocab(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def predict_next_token(input_seq, model, token_to_idx, idx_to_token):
    # Convert input to indices
    input_indices = [token_to_idx[token] for token in input_seq]
    # print(input_indices)
    input_indices = np.array(input_indices).reshape(1, -1)
    # print(input_indices.shape)

    # Predict next token
    prediction = model.predict(input_indices, verbose=0)
    next_token_idx = np.argmax(prediction)
    return idx_to_token[next_token_idx]

def tokenize_input(code, vocab):
    # Use regex to split the input into tokens (consistent with earlier steps)
    tokens = re.findall(r'\b\w+\b|[^\s\w]', code)
    return [token for token in tokens if token in vocab]

def predict_next_word_custom(model, content, thisvocab):
    tokens = tokenize.generate_tokens(iter(content.splitlines()).__next__)
    token_list = []
    for token in tokens:
        if token.type not in (tokenize.COMMENT, tokenize.NL):
            if token.string == '':
                continue
            token_list.append(token.string)
    # print(token_list.shape)
    # print(len(token_list))
    
    for i in range(len(token_list)):
        if thisvocab.get(token_list[i])==None:
            token_list[i] = 1
        else:
            token_list[i] = thisvocab[token_list[i]]
    token_list = np.array(token_list).reshape(1, -1)
    # print(token_list)
    prediction = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(prediction)
    # print(predicted_word_index)
    flag = False
    for key, value in thisvocab.items():
        if value == predicted_word_index:
            flag = True
            return key
    if(flag == False):
        return "No word found"



# Streamlit App
st.title("Code Completion Model")

# Input Text Area
input_text = st.text_area("Enter Code Snippet:", height=100)

# Predict Button
if st.button("Predict Next Token"):
    model_lstm1 = load_lstm_model("LSTM_small_10e.h5")
    model = load_lstm_model("LSTM_big_25e.h5")
    model_gru = load_lstm_model("GRU_big_25.h5")
    # token_to_idx = load_vocab("t_to_i.pkl")
    # idx_to_token = load_vocab("i_to_t.pkl")
    vocab = load_vocab("vocab1.pkl")
    vocab_gru = load_vocab("vocab_gru.pkl")
    if input_text.strip():
        # print(f"Input text:{input_text}")
        max_sequence_len=4
        next_word1=predict_next_word_custom(model,input_text,vocab)
        st.success(f"Predicted Next Token from Model 1:\t {next_word1}")

        # print(list(token_to_idx.keys()))
        input_seq = tokenize_input(input_text, vocab1)
        if len(input_seq) == 0:
            st.error("Predicted Next Token from Model 2:\t No suggestions for the given code snippet!")
        else:
            next_word2=predict_next_token(input_seq, model_lstm1, token_to_idx, idx_to_token)
            st.success(f"Predicted Next Token from Model 2:\t {next_word2}")

        next_word3 = predict_next_word_custom(model_gru,input_text,vocab_gru)
        st.success(f"Predicted Next Token from Model 3:\t {next_word3}")
    else:
        st.error("Please enter some code snippet!")
    
