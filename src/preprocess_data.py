import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    """
    Converte as strings de datas em tensores (numéricos e one-hot)
    com os tamanhos exatos exigidos pela rede neural.
    """
    X_numerico = []
    Y_numerico = []
    
    for data_h, data_m in dataset:
        # Pega o índice do caractere. Se não existir, pega o índice de '<unk>' (unknown) ou 0
        x_indices = [human_vocab.get(char.lower(), human_vocab.get('<unk>', 0)) for char in data_h]
        y_indices = [machine_vocab.get(char, 0) for char in data_m]
        
        X_numerico.append(x_indices)
        Y_numerico.append(y_indices)
    

    # Transforma listas em matrizes de tamanho (m, Tx) e (m, Ty)
    pad_char = human_vocab.get('<pad>', 36)
    X = pad_sequences(X_numerico, maxlen=Tx, padding='post', value=pad_char)
    Y = pad_sequences(Y_numerico, maxlen=Ty, padding='post', value=0)
    

    tamanho_vocab_humano = len(human_vocab)
    tamanho_vocab_maquina = len(machine_vocab)
    #Crias os vetores de one-hot
    Xoh = np.array([to_categorical(i, num_classes=tamanho_vocab_humano) for i in X])
    Yoh = np.array([to_categorical(i, num_classes=tamanho_vocab_maquina) for i in Y])
    
    return X, Y, Xoh, Yoh