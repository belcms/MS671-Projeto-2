import numpy as np 
import tensorflow as tf
from keras.layers import Input, Bidirectional, LSTM, Embedding
from keras.layers import RepeatVector, Concatenate, Dense, Activation, Dot
import pickle
import pandas as pd
from load_dataset import load_dataset
from plot_attention_map import plot_attention_map
from preprocess_data import preprocess_data
from keras.models import load_model, Model
from keras.optimizers import Adam
from string_to_int import string_to_int
from keras.utils import to_categorical
from keras.layers import Softmax
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30 #length da entrada x 
Ty = 10 #length da saída y

X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)


repetidor = RepeatVector(Tx)
concatenador = Concatenate(axis=-1)
densa_1 = Dense(10, activation="tanh") 
densa_2 = Dense(1, activation="relu") 
ativacao_softmax = Softmax(axis=1, name='attention_weights') 
calculo_contexto = Dot(axes=1)


n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

post_activation_LSTM_cell = LSTM(n_s, return_state = True) # Please do not modify this global variable.
output_layer = Dense(len(machine_vocab), activation="softmax")


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

    alphas = ativacao_softmax(energias)
    
    contexto = calculo_contexto([alphas, a])
    
    return contexto


def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):

        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = um_passo_de_atencao(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = post_activation_LSTM_cell(context,initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs=[X, s0, c0],outputs=outputs)

    return model

model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model.summary()

opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, weight_decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']*10)

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

# model.fit([Xoh, s0, c0], outputs, epochs=100, batch_size=100)
# model.save_weights("pesos100epocas.weights.h5")


# model.load_weights('weights/model.h5')
model.load_weights('pesos100epocas.weights.h5')

def translate_date(sentence):
    s00 = np.zeros((1, n_s))
    c00 = np.zeros((1, n_s))
    
    source = string_to_int(sentence, Tx, human_vocab)
    
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    
    prediction = model.predict([source, s00, c00])
    
    indices_vencedores = [np.argmax(p) for p in prediction]
    
    output = [inv_machine_vocab[i] for i in indices_vencedores]
    
    print("source:", sentence)
    print("output:", ''.join(output),"\n")

example = "4th of july 2001"
translate_date(example)

model.summary()

attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);
