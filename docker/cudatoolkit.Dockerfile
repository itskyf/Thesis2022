FROM mambaorg/micromamba:latest as builder

# Split PyTorch to use with runtime
RUN micromamba install -y -n base -c pytorch -c conda-forge \
	cudatoolkit=11.3 pytorch python=3.9 &&\
	micromamba clean -y --all

RUN micromamba install -y -n base -c nvidia/label/cuda-11.3.1 -c conda-forge \
	cuda-toolkit gxx=9 git ninja pip setuptools scikit-image tqdm &&\
	micromamba clean -y --all

RUN micromamba install -y -n base -c conda-forge git rclone tmux vim &&\
	micromamba clean -y --all

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir --upgrade spconv-cu113

USER root
RUN --mount=target=/tmp/context,source=. \
	install -m755 /tmp/context/start.sh ~ &&\
	apt-get update &&\
	apt-get install -y --no-install-recommends openssh-server &&\
	echo 'set -g mouse on' >> ~/.tmux.conf &&\
	echo 'setw -g mode-keys vi' >> ~/.tmux.conf &&\
	echo 'bind Escape copy-mode' >> ~/.tmux.conf &&\
	rm -rf /var/lib/apt/lists/*

CMD ["~/start.sh"]
