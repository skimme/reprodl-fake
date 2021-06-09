FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

COPY . /reprodl
WORKDIR /reprodl

RUN apt update && apt upgrade -y
RUN pip install -r requirements.txt
RUN apt install -y libsndfile1-dev