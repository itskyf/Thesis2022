FROM kerry347/thesis2022:spconv113

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install --yes --channel conda-forge git gxx=10 libxml2 wget &&\
	LD_LIBRARY_PATH=/opt/conda/lib \
	micromamba install --yes --channel conda-forge cudatoolkit-dev=11.3 &&\
	git clone --depth 1 https://github.com/ncakhoa/Thesis2022.git ~/Thesis2022 &&\
	TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6" \
	pip install --no-build-isolation --no-cache-dir --upgrade --editable ~/Thesis2022 &&\
	micromamba remove --yes cudatoolkit-dev gxx libxml2 wget &&\
	micromamba clean --yes --all
