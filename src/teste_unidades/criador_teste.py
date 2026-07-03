from faker import Faker
import random
import csv
from babel.dates import format_date

# Define uma semente para reproducibilidade (opcional)
fake = Faker()
Faker.seed(42)
random.seed(42)

# Lista de formatos suportados pelo Babel e variações personalizadas
FORMATS = [
    'short',
    'medium',
    'long',
    'full',
    'full',
    'full',
    'd MMM YYY', 
    'd MMMM YYY',
    'dd MMM YYY',
    'd MMM, YYY',
    'd MMMM, YYY',
    'dd, MMM YYY',
    'd MM YY',
    'd MMMM YYY',
    'MMMM d YYY',
    'MMMM d, YYY',
    'dd.MM.YY'
]

def generate_date_dataset(num_examples=500):
    """
    Gera um dataset de datas traduzindo formatos legíveis por humanos para o formato YYYY-MM-DD.
    """
    dataset = []
    
    for _ in range(num_examples):
        # 1. Gera uma data aleatória usando o Faker
        dt = fake.date_object()

        # 2. Escolhe um formato de data aleatório da lista
        format_str = random.choice(FORMATS)

        try:
            # 3. Formata a data para a linguagem humana usando o Babel (em inglês)
            human_readable = format_date(dt, format=format_str, locale='en_US')
            
            # (Opcional) Introduz ruídos clássicos do dataset original do Coursera:
            # Mistura minúsculas, remove vírgulas, etc.
            human_readable = human_readable.lower()
            human_readable = human_readable.replace(',', '')
            
            # 4. Formata a mesma data para a máquina (target YYYY-MM-DD)
            machine_readable = dt.isoformat()

            dataset.append((human_readable, machine_readable))
        
        except AttributeError:
            # Ignora eventuais falhas de formatação em datas extremas
            pass
            
    return dataset

# --- Execução do Gerador ---
if __name__ == "__main__":
    # Gera os 500 exemplos solicitados
    test_dataset = generate_date_dataset(500)
    
    # Define o nome do arquivo de saída
    nome_arquivo = "test_dataset.csv"
    
    # Cria e escreve os dados no arquivo
    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escreve o cabeçalho (opcional, mas recomendado)
        writer.writerow(['human_readable', 'machine_readable'])
        
        # Escreve todas as 500 linhas geradas
        writer.writerows(test_dataset)
        
    print(f"Sucesso! O arquivo de saída '{nome_arquivo}' foi salvo na pasta atual.")