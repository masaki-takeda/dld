# docker run -it deep_lab
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
ARG PYTHON_VERSION=3.7

RUN apt-get update
RUN apt-get install -y wget

RUN pip install -U pip

RUN pip install numpy==1.17.4
RUN pip install matplotlib==3.2.1
RUN pip install tqdm==4.36.1
RUN pip install scipy==1.4.1
RUN pip install nibabel==3.0.2
RUN pip install pandas==1.0.3
RUN pip install torch==1.4.0
RUN pip install torchvision==0.5.0
RUN pip install tensorboardX==2.0
RUN pip install h5py==3.2.1
RUN pip install jupyterlab==3.0.16
RUN pip install tensorboard==2.6.0
