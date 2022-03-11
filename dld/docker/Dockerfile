# docker run -it deep_lab
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
ARG PYTHON_VERSION=3.7

RUN apt-get update
RUN apt-get install -y wget

RUN pip install -U pip

RUN pip install numpy matplotlib tqdm
RUN pip install scipy nibabel pandas
RUN pip install torch torchvision
RUN pip install tensorboardX
RUN pip install h5py
RUN pip install jupyterlab
RUN pip install tensorboard==2.6.0
