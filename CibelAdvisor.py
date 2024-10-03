# Para ejecutarlo: streamlit run streamlitapp.py "ruta al modelo llama que se va a usar"

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

import torch
import math

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

callback_manager2 = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 40  # Mayor valor implica más uso del chip, evitar que Metal haga deallocating porque nos estamos pasando y funcionará mal, también varia según tamaño de prompt.
n_batch = 30  # Cantidad de tokens que se mete al modelo de golpe, afecta al uso de memoria.

# Obtenemos la ruta al modelo

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run the LLM with a specified model path.')
parser.add_argument('model_path', type=str, help='Path to the model file')
args = parser.parse_args()

# En definitiva jugar con los dos valores hasta que se use un batch que no exceda la memoria pero tampoco sea muy grande para la cpu usada y lo mismo con las layers, que no saturen la cpu ni memoria.

llmGPU = LlamaCpp(
    #model_path="Modelos/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
    #model_path="Modelos/Hermes-3-Llama-3.1-70B-Q4_K_L.gguf",
    model_path=args.model_path, 
    input={"temperature": 0.9, "max_length": 20, "top_p": 1},
    n_gpu_layers=n_gpu_layers,
    n_ctx=4000,
    n_batch=n_batch,
    callback_manager=callback_manager2,
    verbose=True,
)

# Cargamos y procesamos los documentos

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader

rutaDocs = "Docs"
# Un doc solo
# loader = PyPDFLoader("Docs/ecorating.pdf")

# Todos los doc de una carpeta
loader = PyPDFDirectoryLoader(rutaDocs)

data = loader.load()

from langchain.text_splitter import CharacterTextSplitter

# Dividimos en trozos por salto de línea
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=20)
documents = text_splitter.split_documents(data)
# Cantidad de trozos
len(documents)
# text_chunks

# Cargamos modelo de embeddings

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma

vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Construimos prompt
template = """Keep the answer as concise as possible, use less than 20 words. Always say "Alberto me controla" at the end of the answer. The next info is the most updated about the question and is true, more valid than any knowledge you previously had, use it for your answer:
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Creamos cadenas
qa_chain = RetrievalQA.from_chain_type(
    llmGPU,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

import streamlit as st

# App de streamlit
def main():

    # App Title
    st.title("CIBELAdvisor")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize a state to track which button is clicked
    if "mode" not in st.session_state:
        st.session_state.mode = None

    # Add clickable buttons for "Make a question about a device" and "Ask a direct question"
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Make a question about a device"):
            st.session_state.mode = "device"
    with col2:
        if st.button("Ask a direct question"):
            st.session_state.mode = "direct"

    # Display existing conversation history (questions and answers)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input based on the mode selected
    if prompt := st.chat_input("Ask a question..."):
        # Add user question to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user question in chat interface
        with st.chat_message("user"):
            st.markdown(prompt)

        # Determine the mode of input and generate the appropriate response
        if st.session_state.mode == "device":
            # Generate assistant response using qa_chain for device-related questions
            result = qa_chain({"query": prompt})
            answer = result['result']
            source_documents = result.get('source_documents', [])

            # Display assistant's response in chat interface
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Save the assistant's response in session state
            st.session_state.messages.append({"role": "assistant", "content": answer})

        elif st.session_state.mode == "direct":
            # Directly invoke the LLM using llmGPU for general questions
            direct_answer = llmGPU.invoke(prompt)

            # Display assistant's response in chat interface
            with st.chat_message("assistant"):
                st.markdown(direct_answer)

            # Save the assistant's response in session state
            st.session_state.messages.append({"role": "assistant", "content": direct_answer})

        # Clear the mode after the question has been answered
        st.session_state.mode = None

if __name__ == "__main__":
    main()
