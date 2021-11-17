FROM python:3.6-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpulse-dev \
    portaudio19-dev \
    swig \
    alsa-utils \
    alsa-tools \
    libasound-dev \
    libportaudio2 \
    libportaudiocpp0

COPY requirements.txt requirements.txt
COPY requirements_training.txt requirements_training.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -r requirements_training.txt