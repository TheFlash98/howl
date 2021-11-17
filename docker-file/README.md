# Using docker
Follow the given steps to make a docker for this projects and run a demo:

1. Clone this repo:
    `git clone https://github.com/TheFlash98/howl.git`
    `cd howl`
2. Create a docker using the `Dockerfile` inside the `docker-file` folder:
    `cd docker-file`
    `docker build --tag howl-docker .`
    `cd ..`
3. Run the `hey_fire_fox.py` script using the docker.
    `docker run -it -v $(pwd):/app/ python-docker python hey_fire_fox.py`
    `docker run --privileged -it -v $(pwd):/app/ python-docker python hey_fire_fox.py`