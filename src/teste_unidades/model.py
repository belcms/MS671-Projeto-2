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


def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    repetidor = RepeatVector(Tx)
    concatenador = Concatenate(axis=-1)
    densa_1 = Dense(10, activation="tanh")
    densa_2 = Dense(1, activation="relu")
    ativacao_softmax = Softmax(axis=1, name='attention_weights')
    calculo_contexto = Dot(axes=1)
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(machine_vocab_size, activation="softmax", name="saida")

    def um_passo_de_atencao(a, s_prev):
        s_prev_repetido = repetidor(s_prev)
        concatenado = concatenador([a, s_prev_repetido])
        e = densa_1(concatenado)
        energias = densa_2(e)
        alphas = ativacao_softmax(energias)
        contexto = calculo_contexto([alphas, a])
        return contexto

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s, c = s0, c0
    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    for t in range(Ty):
        context = um_passo_de_atencao(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    return Model(inputs=[X, s0, c0], outputs=outputs)


def translate_date(sentence, model, n_s, Tx, human_vocab, inv_machine_vocab):
    s00 = np.zeros((1, n_s))
    c00 = np.zeros((1, n_s))
    source = string_to_int(sentence, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s00, c00], verbose=0)
    output = ''.join(inv_machine_vocab[np.argmax(p)] for p in prediction)
    print("source:", sentence)
    print("output:", output, "\n")
    return output


def treinar_e_avaliar(n_a, n_s, m, Tx, Ty, human_vocab, machine_vocab, Xoh, Yoh, epochs=30):
    model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, weight_decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'] * 10)

    s0_train = np.zeros((m, n_s))
    c0_train = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    history = model.fit([Xoh, s0_train, c0_train], outputs, epochs=epochs, batch_size=100, verbose=1)
    model.save_weights(f"pesos_na{n_a}_ns{n_s}.weights.h5")

    acc_keys = [k for k in history.history.keys() if k.endswith('accuracy')]
    acc_media = np.mean([history.history[k][-1] for k in acc_keys])
    return model, history, acc_media


# ---------------------------------------------------------------------------
# Tudo abaixo SÓ roda quando você executa "python model.py" diretamente.
# Ao fazer "from model import modelf" em outro script, nada disso é executado.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    combinacoes = [
        (16, 8),
        (32, 16),
        (16,16),
        (8,8),
        (4,4)
    ]

    exemplos_teste = [
        "3 May 1979",
        "Tuesday 09 Oct 1993",
        "5th of March 2021",
        "21 Jun 2005",
        "March 3 2001",
    ]

    m = 10000
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

    Tx = 30  # length da entrada x
    Ty = 10  # length da saída y

    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    lstm_units = 32  # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
    n_s = 64  # number of units for the post-attention LSTM's hidden state "s"

    model = modelf(Tx, Ty, lstm_units, n_s, len(human_vocab), len(machine_vocab))
    # model.summary()

    opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, weight_decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'] * 10)

    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    # model.fit([Xoh, s0, c0], outputs, epochs=100, batch_size=100)
    # model.save_weights("pesos100epocas.weights.h5")

    model.load_weights("pesos100epocas.weights.h5")

    resultados = []
    for n_a, n_s_combo in combinacoes:
        model_atual, history, acc_media = treinar_e_avaliar(
            n_a, n_s_combo, m, Tx, Ty, human_vocab, machine_vocab, Xoh, Yoh, epochs=30
        )
        for data in exemplos_teste:
            traduzido = translate_date(data, model_atual, n_s_combo, Tx, human_vocab, inv_machine_vocab)
            resultados.append({
                "n_a": n_a, "n_s": n_s_combo,
                "exemplo": data, "saida": traduzido,
                "acc_treino_media": acc_media
            })

    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)

    example = "4th of july 2001"
    translate_date(example, model, n_s, Tx, human_vocab, inv_machine_vocab)  # model e n_s globais, intactos (32/64, pesos100epocas)
    model.summary()
    attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num=7, n_s=64)