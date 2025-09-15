# RAGent

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-00B86B?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjMDBiODZiIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTUgMTMuNXYtMy41aDEwVjEzaC0xMHptMCA1di0zLjVoMTB2My41aC0xMHptMTAtMTBoLTl2LTNoOSB2M3oiLz48L3N2Zz4=)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FFD700?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkZEMzAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTExLjUgM2MuODMgMCAxLjUgLjY3IDEuNSAxLjUgMCAuODMtLjY3IDEuNS0xLjUgMS41LS44MyAwLTEuNS0uNjctMS41LTEuNSAwLS44My42Ny0xLjUgMS41LTEuNXpNNSA5Yy44MyAwIDEuNS42NyAxLjUgMS41IDAgLjgzLS42NyAxLjUtMS41IDEuNS0uODMgMC0xLjUtLjY3LTEuNS0xLjUgMC0uODMuNjctMS41IDEuNS0xLjV6bTAgNWMuODMgMCAxLjUuNjcgMS41IDEuNSAwIC44My0uNjcgMS41LTEuNSAxLjUtLjgzIDAtMS41LS42Ny0xLjUtMS41IDAtLjgzLjY3LTEuNSAxLjUtMS41ek0xOSA5Yy44MyAwIDEuNS42NyAxLjUgMS41IDAgLjgzLS42NyAxLjUtMS41IDEuNS0uODMgMC0xLjUtLjY3LTEuNS0xLjUgMC0uODMuNjctMS41IDEuNS0xLjV6bTAgNWMuODMgMCAxLjUuNjcgMS41IDEuNSAwIC44My0uNjcgMS41LTEuNSAxLjUtLjgzIDAtMS41LS42Ny0xLjUtMS41IDAtLjgzLjY3LTEuNSAxLjUtMS41eiIvPjwvc3ZnPg==)](https://www.trychroma.com/)
[![PyPDF2](https://img.shields.io/badge/PyPDF2-008080?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypi.org/project/PyPDF2/)
[![python-docx](https://img.shields.io/badge/python--docx-0078D4?style=for-the-badge&logo=microsoftword&logoColor=white)](https://pypi.org/project/python-docx/)
[![tiktoken](https://img.shields.io/badge/tiktoken-FF4500?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkY0NTAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTUgMTMuNXYtMy41aDEwVjEzaC0xMHptMCA1di0zLjVoMTB2My41aC0xMHptMTAtMTBoLTl2LTNoOSB2M3oiLz48L3N2Zz4=)](https://github.com/openai/tiktoken)
[![marker-pdf](https://img.shields.io/badge/marker--pdf-FFB300?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://pypi.org/project/marker-pdf/)
[![dotenv](https://img.shields.io/badge/dotenv-10AA50?style=for-the-badge&logo=dotenv&logoColor=white)](https://pypi.org/project/python-dotenv/)

RAGent es un asistente conversacional basado en RAG (Retrieval-Augmented Generation) que responde preguntas utilizando información extraída de documentos (PDF, DOCX, TXT) y modelos de lenguaje (LLM). El sistema ingiere archivos, los procesa en chunks, genera embeddings, almacena los vectores en una base ChromaDB y utiliza un modelo LLM para responder preguntas apoyándose en el contexto recuperado.

## Flujo de datos

### Ingesta de archivos

1. El usuario ingresa archivos mediante la CLI ([main.py](main.py), [ingestion.py](app/data/ingestion.py)).

2. Los archivos se leen y procesan (PDF, DOCX, TXT).

3. El texto se divide en chunks ([chunking.py](app/data/chunking.py)).

4. Se generan embeddings para cada chunk ([embeddings.py](app/models/embeddings.py)).

5. Los chunks y sus embeddings se almacenan en ChromaDB.

### Recuperación y respuesta

1. El usuario realiza una consulta en el chat.

2. El sistema recupera los chunks más relevantes desde ChromaDB usando embeddings ([retriever.py](app/rag/retriever.py)).

3. Se construye un prompt con el contexto recuperado y la pregunta del usuario ([qa.py](app/rag/qa.py)).

4. El prompt se envía al modelo LLM para generar una respuesta ([llm.py](app/models/llm.py)).

5. Se muestra la respuesta y las fuentes relevantes al usuario.

### Gestión de la conversación

1. Se mantiene un historial de turnos (usuario/asistente) ([conversation.py](app/controllers/conversation.py), [chatbot.py](app/chatbot.py)).

2. Se puede alternar entre modo RAG y modo LLM puro.

## Diagrama de flujo de datos

```mermaid
flowchart TD
    A[Usuario CLI] -->|Ingesta| B[Lectura de archivos: PDF, DOCX, TXT]
    B --> C[Chunking de texto]
    C --> D[Generación de embeddings]
    D --> E[Almacenamiento en ChromaDB]

    A2[Usuario Chat] -->|Consulta| F[Recuperación de chunks relevantes]
    F --> G[Construcción de prompt]
    G --> H[LLM: OpenAI]
    H --> I[Respuesta y fuentes]

    E --> F

```

## Componentes principales

- [main.py](main.py): CLI para ingesta y ejecución.
- [app/data/ingestion.py](app/data/ingestion.py): Procesamiento de archivos y chunks.
- [app/models/embeddings.py](app/models/embeddings.py): Generación de embeddings.
- [app/rag/retriever.py](app/rag/retriever.py): Recuperación de contexto relevante.
- [app/rag/qa.py](app/rag/qa.py): Construcción de prompts y respuestas.
- [app/models/llm.py](app/models/llm.py): Interfaz con el modelo LLM.
- [app/controllers/conversation.py](app/controllers/conversation.py): Gestión del historial conversacional.
- [app/chatbot.py](app/chatbot.py): Interfaz de chat.

## Requisitos

Consulta el archivo [requirements.txt](requirements.txt) para dependencias necesarias.

## Ejecución

1. Instala las dependencias: `pip install -r requirements.txt`

2. Configura tu archivo [.env](.env) con las claves necesarias (OpenAI, etc).

3. Ingresa documentos usando la CLI: `python main.py ingest files/mi_documento.pdf`

4. Inicia el chat: `python main.py run`