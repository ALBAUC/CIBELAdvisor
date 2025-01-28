# Utiliza como base Miniconda (o Anaconda)
FROM continuumio/anaconda3

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copiamos el archivo de entorno a una carpeta temporal
COPY CondaEnvs/Gemini-env.yml /tmp/Gemini-env.yml

# Creamos el entorno
RUN conda env create -f /tmp/Gemini-env.yml

# Ajustamos la variable PATH para que use el entorno "Gemini-env"
# (Comprueba que el 'name' del .yml sea exactamente 'Gemini-env', en mayúsculas/minúsculas)
ENV PATH /opt/conda/envs/Gemini-env/bin:$PATH

# Copiamos todo tu proyecto a /app
COPY . /app

# Exponemos el puerto que usará streamlit
EXPOSE 8501

# Ejecutamos el script de Streamlit
CMD ["streamlit", "run", "streamlitappV4_Gemini.py"]

