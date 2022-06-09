FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel as builder

ARG CUDA_ARCH
RUN conda install -y --channel conda-forge git ninja pip setuptools &&\
	git clone https://github.com/ncakhoa/Thesis2022.git &&\
	TORCH_CUDA_ARCH_LIST=$CUDA_ARCH pip install \
	--no-build-isolation --no-cache-dir --no-deps --upgrade --editable ./Thesis2022 &&\
	rm -rf ./Thesis2022/build &&\
	conda clean -y --all

FROM mambaorg/micromamba:latest

RUN micromamba install -y --name base --channel pytorch --channel conda-forge \
	cudatoolkit=11.3 numpy python=3.8 pytorch &&\
	micromamba clean -y --all

RUN micromamba install -y --name base --channel conda-forge \
	fire kornia lark pip portalocker pybind11 \
	easydict numba scikit-image sharedarray tensorboard tqdm &&\
	micromamba install -y --name base --channel conda-forge gdown git rclone tmux vim &&\
	micromamba shell init --shell=bash --prefix=$MAMBA_ROOT_PREFIX &&\
	micromamba clean -y --all

USER root
RUN --mount=target=/tmp/context,source=. \
	install -m755 /tmp/context/start.sh ~ &&\
	apt-get update &&\
	apt-get install -y --no-install-recommends openssh-server &&\
	rm -rf /var/lib/apt/lists/*

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN --mount=target=/tmp/builder,source=/workspace,from=builder \
	cp -r /tmp/builder/Thesis2022 ~ &&\
	echo "$HOME/Thesis2022" > $MAMBA_ROOT_PREFIX/lib/python3.8/site-packages/easy-install.pth &&\
	printf "$HOME/Thesis2022\n." > $MAMBA_ROOT_PREFIX/lib/python3.8/site-packages/pcdet.egg-link &&\
	pip install --no-cache-dir --upgrade spconv-cu113

WORKDIR /root/Thesis2022/tools
CMD ["~/start.sh"]
