import pickle

def load_dataset(m):
    """
    Carrega e retorna um subconjunto dos dados e vocabulários do projeto.

    A função abre os arquivos estáticos '.pkl' localizados na pasta 'data/', 
    lê o dataset principal e os dicionários de mapeamento, e retorna
    o dataset limitado aos primeiros 'm' exemplos.

    Args:
        m (int): O número de exemplos do dataset a serem carregados. 
                 Útil para delimitar um subconjunto menor durante testes rápidos.
    
    Returns:
        tuple: Uma tupla contendo 4 elementos:
            - dataset (list): Lista com os primeiros 'm' pares de datas (formato humano, formato de máquina).
            - human_vocab (dict): Dicionário que mapeia os caracteres das datas humanas para índices numéricos.
            - machine_vocab (dict): Dicionário que mapeia os caracteres das datas de máquina para índices numéricos.
            - inv_machine_vocab (dict): Dicionário reverso que mapeia os índices de volta para os caracteres de máquina.
    """
    caminho_dataset = "data/dataset.pkl"
    caminho_human_vocab = "data/human_vocab.pkl"
    caminho_machine_vocab = "data/machine_vocab.pkl"
    caminho_inv_machine_vocab = "data/inv_machine_vocab.pkl"

    with open(caminho_dataset, "rb") as f:
        dataset = pickle.load(f)
    dataset = dataset[:m]
    # print(dataset[:10])

    with open(caminho_human_vocab, "rb") as f:
        human_vocab = pickle.load(f)
    # print(human_vocab)

    with open(caminho_machine_vocab, "rb") as f:
        machine_vocab = pickle.load(f)
    # print(human_vocab)

    with open(caminho_inv_machine_vocab, "rb") as f:
        inv_machine_vocab = pickle.load(f)
    #print(inv_machine_vocab)

    return dataset, human_vocab, machine_vocab, inv_machine_vocab