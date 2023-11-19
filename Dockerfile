# Use an official Python runtime as a parent image
# FROM python:3.9-slim
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
RUN pip install \
    async-timeout==4.0.2 \
    Cython==0.29.32 \
    h5py==3.6.0 \
    huggingface-hub==0.5.1 \
    IProgress==0.4 \
    ipykernel==6.13.0 \
    ipython==7.29.0 \
    ipython-genutils==0.2.0 \
    ipywidgets==7.7.0 \
    joblib==1.1.0 \
    jupyter-client==7.3.4 \
    jupyter-core==4.10.0 \
    jupyter-server==1.17.0 \
    jupyterlab==3.4.0 \
    jupyterlab-pygments==0.2.2 \
    jupyterlab-server==2.13.0 \
    jupyterlab-widgets==1.1.0 \
    matplotlib==3.5.2 \
    matplotlib-inline==0.1.2 \
    mosestokenizer==1.2.1 \
    multidict==6.0.2 \
    nltk==3.7 \
    numba==0.56.4 \
    numpy==1.21.2 \
    pandas==1.3.5 \
    pickleshare==0.7.5 \
    Pillow==8.4.0 \
    pytorch-lightning==1.6.3 \
    pytorch-memlab==0.2.4 \
    pytorch-nlp==0.5.0 \
    requests==2.25.1 \
    scikit-learn==1.0.2 \
    scipy==1.7.3 \
    seaborn==0.11.2 \
    sentencepiece==0.1.97 \
    six==1.16.0 \
    smart-open==5.2.1 \
    tensorboard==2.9.0 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.1 \
    tokenizers==0.12.1 \
    toolwrapper==2.1.0 \
    torch==1.13.0 \
    torchaudio==0.13.0 \
    torchelastic==0.2.0 \
    torchmetrics==0.8.2 \
    torchtext==0.11.0 \
    torchvision==0.11.1 \
    tqdm==4.61.2 \
    transformers==4.18.0

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Specify the default command to run on container start
CMD ["bash"]