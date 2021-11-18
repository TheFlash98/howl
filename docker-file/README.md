# Using docker
Follow the given steps to make a docker for this projects and run a demo:

1. Clone this project.

    `git clone https://github.com/TheFlash98/howl.git`

    `cd howl`
2. Create a docker using the `Dockerfile` inside the `docker-file` folder.

    `cd docker-file`

    `docker build --tag howl-docker .`

    `cd ..`

3. Run the `hey_fire_fox.py` script using the docker.

    `docker run -it -v $(pwd):/app/ python-docker python hey_fire_fox.py`
    
    `docker run --privileged -it -v $(pwd):/app/ python-docker python hey_fire_fox.py`

# Without Docker

1. Install some necessary libraries

    `sudo apt-get -y install libpulse-dev`

    `sudo apt-get -y install portaudio19-dev`

    `sudo apt-get -y install swig`

2. Create a virtual environment and install all the python packages needed

    `python3 -m venv ./howl-venv`

    `source howl-vevn/bin/activate`

    `pip install -r requirements.txt -r requirements_training.txt`

3. Test everything is working by running inferencing for hey fire fox:

    `python hey_fire_fox.py`
