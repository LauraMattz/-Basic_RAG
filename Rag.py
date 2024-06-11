# Importação de pacotes necessários para o funcionamento do código
!pip install faiss-cpu==1.7.4 mistralai  # Instalação dos pacotes faiss para operações de vetor e mistralai para NLP

from mistralai.client import MistralClient  # Cliente API para usar os modelos do Mistral
from mistralai.models.chat_completion import ChatMessage  # Modelo para criar mensagens de chat
import requests  # Módulo para realizar requisições HTTP
import numpy as np  # Biblioteca para manipulação de arrays
import faiss  # Biblioteca para eficiência em busca e agrupamento em grandes conjuntos de vetores
import os  # Módulo para interação com o sistema operacional
from getpass import getpass  # Função para entrada segura de senha

# Solicitação e armazenamento da chave da API para uso no cliente Mistral
api_key = getpass("Type your API Key")  # Solicita ao usuário digitar a chave da API de forma segura
client = MistralClient(api_key=api_key)  # Cria um cliente Mistral usando a chave da API

# Obtenção do texto do arquivo armazenado no GitHub via requisição GET
response = requests.get('https://github.com/LauraMattz/RAG_Bot/blob/main/Marinheiros.txt')
text = response.text  # Extrai o texto da resposta da requisição

# Armazenamento do texto em um arquivo local
f = open('Marinheiros.txt', 'w')  # Abre (ou cria) o arquivo em modo de escrita
f.write(text)  # Escreve o texto no arquivo
f.close()  # Fecha o arquivo para liberar recursos

# Exibe o tamanho do texto para verificação
len(text)  # Retorna o número de caracteres no texto

# Divisão do texto em partes (chunks) para processamento
chunk_size = 2048  # Define o tamanho de cada parte
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]  # Cria uma lista de partes
len(chunks)  # Retorna o número de partes criadas

# Função para obter embeddings de texto usando o modelo do Mistral
def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",  # Modelo de embedding
          input=input  # Texto de entrada
      )
    return embeddings_batch_response.data[0].embedding  # Retorna o embedding do texto

# Criação de embeddings para cada parte do texto
text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])  # Array de embeddings
text_embeddings.shape  # Exibe as dimensões do array de embeddings

# Criação de um índice para busca de vetores usando FAISS
d = text_embeddings.shape[1]  # Dimensão dos vetores de embedding
index = faiss.IndexFlatL2(d)  # Cria um índice L2
index.add(text_embeddings)  # Adiciona os embeddings ao índice

# Busca no índice por vetores similares a um vetor de consulta
D, I = index.search(question_embeddings, k=2)  # Realiza a busca
print(I)  # Imprime os índices dos vetores mais próximos
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]  # Recupera os trechos correspondentes aos índices
print(retrieved_chunk)  # Imprime os trechos recuperados

# Combina o contexto e a pergunta em uma prompt e gera uma resposta
prompt = f"""
Informações de contexto estão abaixo.
{retrieved_chunk}
Com base nas informações de contexto e sem conhecimento prévio, responda à consulta.
Consulta: {question}
Resposta:

"""
# Função para enviar a prompt ao modelo Mistral e obter uma resposta
def run_mistral(user_message, model="mistral-medium-latest"):
    messages = [
        ChatMessage(role="user", content=user_message)  # Cria uma mensagem de usuário
    ]
    chat_response = client.chat(
        model=model,  # Modelo de chat
        messages=messages  # Mensagens incluídas na conversa
    )
    return (chat_response.choices[0].message.content)  # Retorna a resposta gerada

run_mistral(prompt)  # Executa a função para gerar a resposta
