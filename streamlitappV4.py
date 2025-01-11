import argparse
import pandas as pd
import torch
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Comprobar si MPS está disponible en nuestra CPU (DEBUG)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

callback_manager2 = CallbackManager([StreamingStdOutCallbackHandler()])

#cantidad de capas del modelo que se cargarán en GPU 
n_gpu_layers = 40  
#Cantidad de muestras que se procesarán simultáneamente en una iteración
n_batch = 30  

#Instanciamos el modelo Llama con LlamaCpp y lo guardamos en el estado de la sesión
if "llmGPU" not in st.session_state:
    #Como se ejcuta un modelo local, obtenemos el path del modelo como argumento
    parser = argparse.ArgumentParser(description='Run the LLM with a specified model path.')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    args = parser.parse_args()

    #instanciamos el modelo Llama con los parámetros que hemos considerado
    st.session_state.llmGPU = LlamaCpp(
        model_path=args.model_path, 
        input={"temperature": 0.60, "max_length": 20, "top_p": 1},
        n_gpu_layers=n_gpu_layers,
        n_ctx=8000,
        n_batch=n_batch,
        callback_manager=callback_manager2,
        verbose=True,
    )

#Modelo de embeddings para tokenizar y vectorizar texto
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #Otra opción de embeddings por si la primera no estuviese disponible
    #st.session_state.embeddings = HuggingFaceEmbeddings(model_name="imone/Llama-3-8B-fixed-special-embedding")

#Función para cargar los documentos y construir el vectorstore
#Pongo el decorador para evitar recargar la aplicación completa para construir una nueva instancia de vectorstore, por el comportamiento de Streamlit
@st.cache_resource
def load_documents_and_build_vectorstore(folder_path):
    #Cargar los documentos del directorio que usaremos para la vectorstore
    try:
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()

        if not documents:
            print("HOLA1")
            return None

        #Los dividimos en fragmentos de texto que se selecciorán cuando se invoque el modelo
        #Separador salto de línea porque se han formateado así los documentos (Cada línea informa de un dispositivo diferente)
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=20)
        split_documents = text_splitter.split_documents(documents)

        if not split_documents:
            print("HOLA2")
            return None

        #Creamos el vectorstore con los documentos divididos
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=st.session_state.embeddings
        )
        return vector_store
    except Exception as e:
        print(f"Error in load_documents_and_build_vectorstore: HOLA 3 {e}")
        return None


#Construimos el prompt para el QA Chain
#Usa el formato necesario para que un modelo Llama Instruct pueda interpretar la información correctamente
QA_CHAIN_PROMPT = PromptTemplate.from_template(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an expert in cybersecurity. I will give you info about a related device. Use that info to answer the question. The info is not necesarilly about the same device.\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "The info I have about a related device is: {context}"
    "My question is: {question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)


#Función principal de la aplicación
#Nota: Hay varias recargas de la aplicación y gestiones de datos recurrentes por la naturaleza de Streamlit
def main():
    #Título de la APP de Streamlit
    st.title("CIBELAdvisor")

    #Inicialización de las variables de estado de la sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "doc_category" not in st.session_state:
        st.session_state.doc_category = None
    if "question_asked" not in st.session_state:
        st.session_state.question_asked = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "selected_devices" not in st.session_state:
        st.session_state.selected_devices = []  #Para almacenar los dispositivos seleccionados por el usuario

    #Barra lateral para seleccionar los dispositivos
    st.sidebar.title("Choose your home devices")
    
    #Lista de dispositivos para seleccionar de los que tenemos información
    devices = [
        "ABUS Secvest FUAA50000", "Alcatel-Lucent CellPipe 7130", "Amazon Echo Dot", "Apple Mac Mini", 
        "Apple Watch Ultra", "ASUS DSL-AC51", "ASUS ROG Rapture GT-AX11000", "ASUS ROG Zephyrus M GM501GS", 
        "ASUS RT-AC1200HP", "ASUS RT-AC68U", "ASUS RT-N56U", "ASUS VivoMini", "ASUS ZenFone 3 Laser", 
        "ASUS ZenFone 3 Max", "Asus ZenFone 3s Max", "Asus ZenFone 5Q", "Asus ZenFone Max 4", 
        "ASUSTek ZenBook Pro Due 15 UX582", "Ayision Ays-WR01", "BRAVIA Smart TV", "Brother MFC-9970CDW", 
        "Chuango 433 MHz burglar-alarm", "Dell 3000cn", "Dell Inspiron 5675", "Dell Latitude 2110", 
        "Dell Latitude 7202", "Dell Latitude E6430", "Dell Latitude Z600", "Dell Optiplex", 
        "Dell Precision 7910", "Dell Precision 7920", "Dell XPS 13 2-in-1", "Dell XPS 13 9370", 
        "Dongguan Diqee Diqee360", "FLIR AX8", "Geutebruck G-Cam/EFD-2250", "Google Home", "HP EliteBook 850", 
        "HP ElitePad 900", "HP OfficeJet Pro", "Huawei HG8247", "Huawei Honor 10", "Huawei Honor 4C", 
        "Huawei Honor 5C", "Huawei Honor 5S", "Huawei Honor 6", "Huawei Honor 7", "Huawei Honor 8 Lite", 
        "Huawei Honor Cube WS860", "Huawei Honor V10", "Huawei Honor V20", "Huawei HONOR 20 PRO", 
        "Huawei Mate 20", "Huawei Mate S", "Huawei P10", "Huawei P10 Plus", "Huawei P9", "Huawei P9 Lite", 
        "iPhone 10", "iPhone 11", "iPhone 12", "iPhone 13", "Iphone 5", "Iphone 7", "Iphone 8", "Iphone 9", 
        "Intelbras NPLUG", "Innominate mGuard Smart HW", "JBL Go 2", "JBL TUNE500BT", "Jisiwei i3", "Lenovo Flex", 
        "Lenovo Miix", "Lenovo ThinkPad A275", "Lenovo ThinkPad A285", "Lenovo ThinkPad A475", 
        "Lenovo ThinkPad A485", "Lenovo ThinkPad T440s", "Lenovo ThinkPad T460p", "Lenovo ThinkPad T495", 
        "Lenovo ThinkPad W541", "Lenovo ThinkPad X395", "Lenovo Yoga", "Microsoft Xbox 360", 
        "Nokelock Smart padlock O1", "Pebble Smartwatch", "Philips Smart Wi-Fi Wiz Connected", 
        "Playstation 3", "Radio Thermostat CT50", "Radio Thermostat CT80", "Rittal Chiller SK 3232-Series", 
        "Samsung 850 Pro", "Samsung A3", "Samsung A5", "Samsung A7", "Samsung A8+", "Samsung D6000 TV", 
        "Samsung Exynos", "Samsung Galaxy Note 2", "Samsung Galaxy S10", "Samsung Galaxy S4", "Samsung Galaxy S5", 
        "Samsung Galaxy S6", "Samsung Galaxy S7", "Samsung Galaxy S9", "Samsung J5", "Samsung J7 Neo", 
        "Samsung Note10", "Samsung PS50C7700", "Samsung S2 Galaxy", "Samsung S7 Edge", "Samsung SHR-5162", 
        "Samsung Smart TV NT14U", "Samsung Smart TV X10P", "Samsung Smart TV X12", "Samsung Smart TV X14H", 
        "Samsung Smart TV X14J", "Seagate ST500LT015", "Seagate ST500LT025", "Smart TV Box H96", 
        "Sophos Cyberoam", "TCL Android Smart TV V8-R851T02-LF1", "Thinkpad X1 Fold Gen 1", "Tinxy Door Lock", 
        "Tollgrade SmartGrid LightHouse Sensor", "TP-LINK ARCHER C50", "TP-Link Tapo C200", "TPLink Tapo L530", 
        "TV Vizio E50x", "TV Vizio P65", "Xiaomi 5S Plus", "Xiaomi Mi A1", "Xiaomi Mi Note 2", "Xiaomi Mi Pad 4", 
        "Xiaomi MIX 2", "Xiaomi MIX 3", "Xiaomi Redmi 6 Pro", "Xiaomi Redmi Note 6 Pro"
    ]



    #Barra de búsqueda para filtrar los dispositivos en la barra lateral
    search_query = st.sidebar.text_input("Search devices", "")

    #Filtrar los dispositivos basados en la consulta de búsqueda
    filtered_devices = [device for device in devices if search_query.lower() in device.lower()]

    #Botón para limpiar los dispositivos seleccionados
    if st.sidebar.button("Clear selected devices"):
        st.session_state.selected_devices.clear()  

    #Mostrar los dispositivos filtrados en la barra lateral
    for device in devices: 
        if device in filtered_devices:  #Solo mostrar los dispositivos que coinciden con la consulta de búsqueda
            is_checked = device in st.session_state.selected_devices
            if st.sidebar.checkbox(device, value=is_checked):
                if device not in st.session_state.selected_devices:
                    st.session_state.selected_devices.append(device)  #Dispositivo seleccionado anhadido
                    st.rerun()
            else:
                if device in st.session_state.selected_devices:
                    st.session_state.selected_devices.remove(device)  #Dispositivo sin seleccionar eliminado
                    st.rerun()

    

    #Mostrar conversación pasada
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Botones para elegir el modo de la aplicación
    if not st.session_state.question_asked:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            #Pregunta sobre un dispositivo que cargará la vectorstore de sostenibilidad o vulnerabilidades con el RAG
            if st.button("Single device question"):
                st.session_state.mode = "device"
                st.session_state.question_asked = True
                st.session_state.messages.append({"role": "assistant", "content": "Alright, choose between Sustainability or Vulnerabilities."})

        with col2:
            #Pregunta directa que se responderá con el LLM
            if st.button("Ask a direct question"):
                st.session_state.messages.append({"role": "assistant", "content": "Ok. Ask me a question about cybersecurity in general."})
                st.session_state.mode = "direct"
                st.session_state.question_asked = True
                st.rerun()
                
        with col3:
            #Evaluar los dispositivos seleccionados en el hogar de la barra lateral
            if st.button("Evaluate my home"):
                selected_devices = st.session_state.selected_devices
                
                if not selected_devices:
                    st.session_state.messages.append({"role": "assistant", "content": "No devices selected."})
                else:
                    st.session_state.mode = "home"
                    st.session_state.question_asked = True
                st.rerun()
 
    #Modo "home" para evaluar los dispositivos seleccionados
    if st.session_state.mode == "home" and st.session_state.selected_devices:
        #Cargamos el archivo CSV con las vulnerabilidades de los dispositivos
        df = pd.read_csv("ListaDispositivosCVE_CWE.csv")
        device_responses = []  #Respuestas para cada dispositivo seleccionado

        #Procesamos cada dispositivo
        for device in st.session_state.selected_devices:
            # Buscamos el dispositivo en modelo
            result = df[df['Modelo'] == device]
            device_data = ""

            #Traemos la información de las vulnerabilidades del dispositivo
            if not result.empty:
                contador_vulnerabilidades = 1
                for _, row in result.iterrows():
                    cve_id = row['CVE ID']
                    cwe_description = row['DescripcionCWE']
                    cve_description = row['English Description']
                    #Para casos sin CWE, usamos el CVE
                    if pd.notna(cwe_description):
                        device_data += f"Vulnerability {contador_vulnerabilidades}: {cve_id} - {cwe_description}\n"
                    else:
                        device_data += f"Vulnerability {contador_vulnerabilidades}: {cve_id} - {cve_description}\n"
                    contador_vulnerabilidades += 1


            #Si no hay información de vulnerabilidades para el dispositivo
            if not device_data:
                device_data = f"No vulnerabilities found for {device}.\n"
            
       #Creamos el prompt para el LLM con los datos del dispositivo
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are an expert in cybersecurity.\n\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                "I will give you the vulnerabilities that my device has in a complex way, but "
                "you will summarize the information and explain it with easy words. "
                "Here is the information for the device:\n"
                f"{device_data}\n\n"
                #"Now, you can give me the summarization:\n\n"
                "Be very BRIEF, dont say aything except the summarization:\n\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            
            #procesamos el prompt con el LLM 
            print(prompt)  # Sacar prompt que se le envía para debug por terminal
            response = st.session_state.llmGPU.invoke(prompt)
            
            #Almacenamos la respuesta para el dispositivo
            device_responses.append(f"Device {device}:\n{response}")

        #Mostramos el conjunto de respuestas para los dispositivos seleccionados
        final_summary = "\n\n".join(device_responses)
        print(final_summary)  # DEBUG

        response = final_summary

        #Mostramos la respuesta en la interfaz de chat
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        #Reseteamos los estados
        st.session_state.mode = None
        st.session_state.question_asked = False
        st.session_state.doc_category = None
        st.rerun()

    #Caso de pregunta sobre un dispositivo en particular
    if st.session_state.mode == "device" and st.session_state.doc_category is None:
        st.write("Please choose between Sustainability or Vulnerabilities:")
        col1, col2 = st.columns(2)

        #Mostramos dos nuevos botones para elegir entre sostenibilidad y vulnerabilidades
        with col1:
            if st.button("Sustainability"):
                #Cargamos los fichero en Eco y construimos la vectorstore
                st.session_state.doc_category = "Eco"
                st.session_state.vector_store = load_documents_and_build_vectorstore("EcoDoc")
                st.session_state.messages.append({"role": "assistant", "content": "Alright, ask me about the sustainability of your device."})
                st.session_state.mode = "waiting"
                st.session_state.question_asked = True
                st.rerun()

        with col2:
            if st.button("Vulnerabilities"):
                #Cargamos los ficheros sobre vulnerabilidades y construimos la vectorstore
                st.session_state.doc_category = "CVEs"
                st.session_state.vector_store = load_documents_and_build_vectorstore("CVEsDoc")
                st.session_state.messages.append({"role": "assistant", "content": "Alright, ask me about the vulnerabilities of your device."})
                st.session_state.mode = "waiting"
                st.session_state.question_asked = True
                st.rerun()

    #Acción cuando se escribe una cuestión en el chat
    if prompt := st.chat_input("Ask a question..."):
        #Guardamos pregunta en el historial
        st.session_state.messages.append({"role": "user", "content": prompt})

        #Mostramos la pregunta en la interfaz de chat
        with st.chat_message("user"):
            st.markdown(prompt)

        #Verificamos si hay algún modo seleccionado, en caso negativo preguntamos directamente
        if st.session_state.mode is None:
            st.session_state.mode = "direct"

        #Caso de pregunta sobre un dispositivo en particular, creamos el QA Chain y respondemos
        if st.session_state.mode == "waiting" and st.session_state.doc_category is not None:  
            qa_chain = RetrievalQA.from_chain_type(
                st.session_state.llmGPU, #Usamos Gemini como modelo de lenguaje
                retriever=st.session_state.vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

            #Ejecutamos el QA Chain y obtenemos la respuesta
            result = qa_chain({"query": prompt})
            answer = result['result']

            #DEBUG
            source_documents = result['source_documents']  #Documentos (Chunks) usados para responder la pregunta


            with st.chat_message("assistant"):
                st.markdown(answer)
            #Mostramos la respuesta en la interfaz de chat
            st.session_state.messages.append({"role": "assistant", "content": answer})

            #DEBUG (Mostrar documentos usados para responder la pregunta)
            print("### Source Documents:")
            for doc in source_documents:
                print(f"**Document Title:** {doc.metadata.get('title', 'Unknown Title')}")
                print(f"**Document Content (snippet):** {doc.page_content[:500]}")

        #Pregunta directa
        elif st.session_state.mode == "direct":
            #Prompt para el LLM
            full_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are an expert in cybersecurity.\n\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}\n\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            direct_answer = st.session_state.llmGPU.invoke(full_prompt)

            with st.chat_message("assistant"):
                st.markdown(direct_answer)

            st.session_state.messages.append({"role": "assistant", "content": direct_answer})

        #Reseteamos los estados
        st.session_state.mode = None
        st.session_state.doc_category = None
        st.session_state.question_asked = False
        st.rerun()

if __name__ == "__main__":
    main()


