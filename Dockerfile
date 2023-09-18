FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /work

COPY ./requirements /work

RUN pip install -r requirements \
    && pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html