# ---- script separado, só avaliação, sem retreinar ----
import os
import numpy as np
import pandas as pd
from load_dataset import load_dataset
from string_to_int import string_to_int
from keras.utils import to_categorical
from model import modelf  
from plot_attention_map import plot_attention_map

# Carrega o dataset original apenas para extrair os vocabulários (human_vocab, machine_vocab)
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx, Ty = 30, 10

combinacoes = [(16, 32), (32, 64), (64, 128), (8, 16), (16, 8), (32, 16), (16, 16), (8, 8), (4, 4), (128, 256)]

def carregar_modelo(n_a, n_s):
    model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.load_weights(f"pesos_na{n_a}_ns{n_s}.weights.h5")
    return model

def acuracia_no_dataset(model, n_s, Xoh, Y, batch_size=512):
    n = Xoh.shape[0]
    s0 = np.zeros((n, n_s))
    c0 = np.zeros((n, n_s))
    predictions = model.predict([Xoh, s0, c0], batch_size=batch_size, verbose=0)
    pred_idx = np.array([np.argmax(p, axis=1) for p in predictions]).T  # (n, Ty)
    acc_char = np.mean(pred_idx == Y)
    acc_seq = np.mean(np.all(pred_idx == Y, axis=1))
    return acc_char, acc_seq

if __name__ == "__main__":
    # 1. Carregar o dataset de teste gerado
    print("Carregando test_dataset.csv...")
    df_test = pd.read_csv("test_dataset.csv")
    X_test_raw = df_test['human_readable'].values
    Y_test_raw = df_test['machine_readable'].values

    # 2. Preprocessar os dados de teste (String -> Int -> One-hot)
    X_test_int = np.array([string_to_int(x, Tx, human_vocab) for x in X_test_raw])
    X_test_oh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X_test_int)))
    
    Y_test_int = np.array([string_to_int(y, Ty, machine_vocab) for y in Y_test_raw])

    # Pega o primeiro exemplo do teste para gerar o mapa de atenção visual
    exemplo_plot = X_test_raw[0]

    # 3. Preparar o arquivo TXT para salvar os resultados
    arquivo_resultados = "resultados_precisao.txt"
    
    with open(arquivo_resultados, "w", encoding="utf-8") as f_out:
        f_out.write("Resultados da Avaliação no test_dataset.csv (500 amostras)\n")
        f_out.write("="*60 + "\n\n")

        # 4. Avaliar cada combinação
        for n_a, n_s in combinacoes:
            print(f"\nAvaliando modelo n_a={n_a}, n_s={n_s}...")
            
            try:
                model = carregar_modelo(n_a, n_s)
                
                # Calcula acurácia com os dados do test_dataset.csv
                acc_char, acc_seq = acuracia_no_dataset(model, n_s, X_test_oh, Y_test_int)
                
                # Grava no TXT
                linha_resultado = f"Modelo (n_a={n_a:3d}, n_s={n_s:3d}) | Acc Char: {acc_char:.4f} | Acc Seq (Frase Completa): {acc_seq:.4f}"
                print(linha_resultado)
                f_out.write(linha_resultado + "\n")

                # Gera o mapa de atenção (a função original salva como 'attention_map.png')
                plot_attention_map(model, human_vocab, inv_machine_vocab, exemplo_plot, n_s=n_s)

                # Renomeia o arquivo salvo para evitar que o próximo loop sobrescreva
                nome_plot_novo = f"attention_map_na{n_a}_ns{n_s}.png"
                if os.path.exists("attention_map.png"):
                    os.replace("attention_map.png", nome_plot_novo)
                    print(f"Gráfico salvo como: {nome_plot_novo}")

            except Exception as e:
                erro_msg = f"Erro ao avaliar modelo n_a={n_a}, n_s={n_s}: {e}"
                print(erro_msg)
                f_out.write(erro_msg + "\n")

    print(f"\nFinalizado! Resultados consolidados salvos em '{arquivo_resultados}'.")