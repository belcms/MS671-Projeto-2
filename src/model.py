import numpy as np 
import tensorflow as tf
from keras.layers import Input, Bidirectional, LSTM, Embedding
from keras.layers import RepeatVector, Concatenate, Dense, Activation, Dot
import pickle


#lê arquivos em data
caminho_dataset = "data/dataset.pkl"
caminho_human_vocab = "data/human_vocab.pkl"
caminho_machine_vocab = "data/machine_vocab.pkl"

with open(caminho_dataset, "rb") as f:
    dataset = pickle.load(f)

with open(caminho_human_vocab, "rb") as f:
    human_vocab = pickle.load(f)

with open(caminho_machine_vocab, "rb") as f:
    machine_vocab = pickle.load(f)

human_vocab_size = len(human_vocab) 

embedding_dim: int = 64
max_sequence_length: int = 0
lstm_units: int = 32


for data_human, data_maq in dataset:
    actual_length = len(data_human)
    if actual_length > max_sequence_length:
        max_sequence_length = actual_length


X_entrada = Input(shape=(max_sequence_length,)) #ver da dimensionalidade

x_emb = Embedding(input_dim=human_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(X_entrada)

a = Bidirectional(LSTM(units=lstm_units, return_sequences= True))(x_emb)
