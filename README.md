# Text Retrieval and Response Generation System 📖🤖

Este projeto implementa um sistema de recuperação de texto e geração de resposta usando embeddings e modelos de processamento de linguagem natural (NLP) fornecidos pelo `mistralai`. O sistema busca trechos de texto relevantes em um documento e utiliza essas informações para responder consultas contextualmente informadas.

## Funcionalidades 🛠️

- **Recuperação de Texto** 📑: Divide um documento em partes e cria embeddings para cada trecho.
- **Busca de Similaridade** 🔍: Utiliza o `faiss` para buscar eficientemente os trechos mais relevantes em resposta a uma consulta.
- **Geração de Resposta** 💬: Combina os trechos recuperados com uma consulta e gera uma resposta usando um modelo de NLP.

## Tecnologias Utilizadas 🧰

- **Faiss**: Para indexação e busca rápida em grandes conjuntos de vetores.
- **MistralAI**: Para geração de embeddings e interação com modelos de chat.
- **Requests**: Para fazer requisições HTTP.
- **Numpy**: Para manipulação de arrays.

## Uso 📌
Antes de executar o script, certifique-se de ter uma chave de API válida para mistralai. Você será solicitado a inserir essa chave ao executar o script.
