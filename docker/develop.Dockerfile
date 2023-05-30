ARG TAG=latest
FROM rtknet:${TAG}

# Install clang-format for linter runs
RUN apt-get update && apt-get install -y -qq --no-install-recommends python3-setuptools

# create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --system --gid ${GROUP_ID} appuser
RUN useradd --create-home --no-log-init --system --uid ${USER_ID} --gid ${GROUP_ID} --groups sudo appuser
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Install flake and black for linting
RUN pip install --user flake8
RUN pip install --user black==22.12.0

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 && \
        cd detectron2 && git reset --hard ecb786ccd26de4086a6fede437e7d728199470e2 && cd .. && \
        pip install --user -e detectron2

# Install RT-K-Net
RUN git clone https://github.com/markusschoen/RT-K-Net.git && \
        pip install --user -e RT-K-Net

WORKDIR /home/appuser/RT-K-Net
