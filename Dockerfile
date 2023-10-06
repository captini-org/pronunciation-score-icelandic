FROM ubuntu:latest

# Install required packages
RUN apt-get update -yqq && apt-get install -y curl bzip2 && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    chmod +x "Miniforge3-$(uname)-$(uname -m).sh" && \
    bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p /root/miniconda3 && \
    rm "Miniforge3-$(uname)-$(uname -m).sh"

# Set environment variables
ENV PATH="/root/miniconda3/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy environment.yml
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml -n captini
# Activate the environment
SHELL ["conda", "run", "-n", "captini", "/bin/bash", "-c"]
RUN apt update
RUN apt install gcc -y
RUN pip install --upgrade pip
RUN pip install pika 
RUN pip install pysoundfile 
RUN pip install scipy 
RUN pip install dtw-python  
RUN pip install transformers
# torch dtw-python transformers pysoundfile scipy
# Copy application files
COPY $RECORDING app/
COPY . .

RUN apt update
RUN apt-get -y update && apt-get install -y libsndfile1
RUN apt install vim -y

# Set entry point
COPY . .
ENV TRANSFORMERS_CACHE=/cache
ENTRYPOINT ["bash", "-c", "conda run -n captini --live-stream python connector.py"]
