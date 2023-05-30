# Installation
## Docker (Recommended)
The recommended way for using RT-K-Net is from within a Docker container.
Only the NVIDIA driver (version ≥ 520), Docker (https://docs.docker.com/get-docker/), and nvidia-docker (https://github.com/NVIDIA/nvidia-docker) have to be installed on your host system for using these images.

We provide two Dockerfiles. A base Dockerfile, which contains all the necessary dependencies to use RT-K-Net, and a develop.Dockerfile, which can be used for adding new features to RT-K-Net. 

To build the base `rtknet:latest` image, run 
```bash
cd docker
./build_docker.sh
```

Optionally, after building `rtknet:latest`, you can build the development container `rtknet-dev:latest` by running `./build_docker.sh -d`.

To run a container, call `./docker/run_docker.sh`, which starts a new container in interactive bash mode. The project source code is located in `/opt/RT-K-Net`.

To run a development container, call `./docker/run_docker.sh -d`.

See `./docker/run_docker.sh -h` for all docker run options.

## Local
### Requirements

* Linux with Python ≥ 3.6
* NVIDIA driver 520
* NVIDIA cuda 11.8.0
* NVIDIA cuDNN 8.6.0.163

### Install RT-K-Net

Call the following commands from the root directory of this repository to install RT-K-Net along with all necessary dependencies.
We recommend using a `virtualenv` for this.

```bash
# System requirements
sudo apt update
sudo apt install -y  \
	build-essential \
	isort \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libcap-dev \
	libgl1-mesa-glx \
	libusb-1.0-0 \
	libglvnd-dev \
	libgl1-mesa-dev \
	libegl1-mesa-dev \
	libx11-6 \
	libgtk2.0-dev

# (Optional) Create and source virtual environment
python3 -m venv venv
source venv/bin/activate

# Python dependencies
python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -r requirements.txt

# Other dependencies
mkdir deps
cd deps
# Install detectron2
git clone https://github.com/facebookresearch/detectron2
python3 -m pip install -e detectron2
cd ..

# Install RT-K-Net in editable mode
python3 -m pip install -e .
```
