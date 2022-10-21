FROM nvidia/cuda:11.0.3-base-ubuntu20.04
RUN apt-get update

ENV CUDA_VERSION=cu102
ENV TORCH_VERSION=1.10.0
ENV ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y install python3.8
RUN apt-get update && apt-get -y install python3-pip --fix-missing

RUN pip install --no-cache-dir torch==$TORCH_VERSION+$CUDA_VERSION torchvision==0.11.0+$CUDA_VERSION torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y \
    wget\
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# All users can use /home/user as their home directory.
RUN mkdir /home/user
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install HDF5 Python bindings.
RUN pip install --no-cache-dir h5py==3.7.0
RUN pip install --no-cache-dir h5py-cache==1.0

# Install TorchNet, a high-level framework for PyTorch.
RUN pip install --no-cache-dir torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests.
RUN pip install --no-cache-dir requests==2.19.1

# Install Graphviz.
RUN pip install --no-cache-dir graphviz==0.20
# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   libgtk2.0-0 \
   libcanberra-gtk-module \
&& sudo rm -rf /var/lib/apt/lists/*


RUN pip --no-cache-dir install scipy==1.8.0
RUN pip install --no-cache-dir --no-index torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html \
 && pip install --no-cache-dir --no-index torch-sparse -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html \
 && pip install --no-cache-dir --no-index torch-cluster -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html \
 && pip install --no-cache-dir --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html \
 && pip install --no-cache-dir git+https://github.com/pyg-team/pytorch_geometric.git\
 && pip install --no-cache-dir pytorch-lightning==1.5.10  transformers==4.19.2\
 && pip install --no-cache-dir --no-deps sentence-transformers==2.2.0 sentencepiece==0.1.96 \
 && pip install --no-cache-dir nltk==3.7 ipykernel==6.13.0 \
 && pip install --no-cache-dir dgl==0.6.1

RUN pip install jupyter
RUN pip install matplotlib

RUN pip install --no-cache-dir "ray[tune]"==1.13.0 \
   bayesian-optimization==1.2.0\
   optuna==2.10.0\
   tensorboardX==2.4.1
# Set the default command to python3.
CMD ["python3"]

#docker rm -f $(docker ps -a -q) && docker volume rm $(docker volume ls -q)