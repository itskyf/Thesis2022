FROM mambaorg/micromamba:latest as builder

# Split PyTorch to use with runtime
RUN micromamba install -y -n base -c pytorch -c conda-forge \
	cudatoolkit=11.6 pytorch python=3.9 &&\
	micromamba clean -y --all

RUN micromamba install -y -n base -c nvidia/label/cuda-11.6.2 -c conda-forge \
	cuda-toolkit gxx=11 git ninja pip setuptools &&\
	micromamba clean -y --all

ARG CUDA_ARCH
RUN cd ~ && git clone --branch recall --single-branch https://github.com/ncakhoa/Thesis2022.git &&\
	TORCH_CUDA_ARCH_LIST=$CUDA_ARCH pip install \
	--no-build-isolation --no-cache-dir --no-deps --upgrade --editable ./Thesis2022 &&\
	rm -rf ./Thesis2022/build

FROM mambaorg/micromamba:latest

RUN micromamba install -y -n base -c pytorch -c conda-forge \
	cudatoolkit=11.6 pytorch python=3.9 &&\
	micromamba clean -y --all

RUN micromamba install -y -n base --c conda-forge \
	easydict numba scikit-image sharedarray tabulate tensorboard tqdm &&\
	micromamba install -y -n base -c conda-forge gdown git rclone tmux vim &&\
	micromamba shell init --shell=bash --prefix=$MAMBA_ROOT_PREFIX &&\
	micromamba clean -y --all

USER root
RUN --mount=target=/tmp/context,source=. \
	install -m755 /tmp/context/start.sh ~ &&\
	apt-get update &&\
	apt-get install -y --no-install-recommends openssh-server &&\
	echo 'set -g mouse on' >> ~/.tmux.conf &&\
	echo 'setw -g mode-keys vi' >> ~/.tmux.conf &&\
	echo 'bind Escape copy-mode' >> ~/.tmux.conf &&\
	rm -rf /var/lib/apt/lists/*

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN --mount=target=/tmp/builder,source=/home/mambauser,from=builder \
	cp -r /tmp/builder/Thesis2022 ~ &&\
	echo "$HOME/Thesis2022" > $MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/easy-install.pth &&\
	printf "$HOME/Thesis2022\n." > $MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/pcdet.egg-link

WORKDIR /root/Thesis2022/tools
CMD ["~/start.sh"]
