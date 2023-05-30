FROM nvcr.io/nvidia/pytorch:22.10-py3

ENV DEBIAN_FRONTEND="noninteractive" \
    TERM="xterm"
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install apt packages
RUN apt-get update && apt-get install -y -qq --no-install-recommends  \
	build-essential \
	isort \
	meld \
	zsh \
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

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Detectron2 env vars
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV FVCORE_CACHE="/tmp"

# Install python dependencies
RUN /opt/conda/bin/pip install pip==22.2.1
RUN /opt/conda/bin/pip install cityscapesscripts==2.2.0
RUN /opt/conda/bin/pip install opencv-python==4.1.2.30  # Newer versions have a bug with imshow calls inside docker
RUN /opt/conda/bin/pip install pickle5==0.0.11
RUN /opt/conda/bin/pip install Shapely==1.8.2
RUN /opt/conda/bin/pip install timm==0.6.13
RUN /opt/conda/bin/pip install gdown==4.7.1
RUN /opt/conda/bin/pip install setuptools==59.5.0  # Newer versions have a bug with torch tensorboard
RUN /opt/conda/bin/pip install git+https://github.com/cocodataset/panopticapi.git

# Install detectron2
RUN cd /opt && \
    git clone https://github.com/facebookresearch/detectron2 && \
    cd detectron2 && git reset --hard ecb786ccd26de4086a6fede437e7d728199470e2 && cd .. && \
    /opt/conda/bin/pip install -e detectron2

# Install RT-K-Net
RUN cd /opt && \
    git clone https://github.com/markusschoen/RT-K-Net.git && \
    /opt/conda/bin/pip install -e RT-K-Net
