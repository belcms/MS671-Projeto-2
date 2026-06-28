import numpy as np 
import tensorflow as tf
from keras.layers import Input, Bidirectional, LSTM, Embedding
from keras.layers import RepeatVector, Concatenate, Dense, Activation, Dot
import pickle
import pandas as pd
from load_dataset import load_dataset


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

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


repetidor = RepeatVector(max_sequence_length)
concatenador = Concatenate(axis=-1)
densa_1 = Dense(10, activation="tanh") 
densa_2 = Dense(1, activation="relu") 
ativacao_softmax = Activation("softmax", name='attention_weights')
calculo_contexto = Dot(axes=1)

def um_passo_de_atencao(a, s_prev):
    """
    Executa um passo de atenção para calcular o vetor de contexto.
    
    a: Saídas escondidas da Bi-LSTM (codificador)
    s_prev: Estado escondido anterior da LSTM (decodificador)
    """

    s_prev_repetido = repetidor(s_prev)
    
    concatenado = concatenador([a, s_prev_repetido])
    
    e = densa_1(concatenado)
    energias = densa_2(e)
    

    alfas = ativacao_softmax(energias)
    
    contexto = calculo_contexto([alfas, a])
    
    return contexto