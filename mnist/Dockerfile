FROM pytorch/pytorch

WORKDIR /workspace

COPY . /workspace

RUN apt-get update
RUN apt-get install -y vim
RUN pip install -r requirements.txt