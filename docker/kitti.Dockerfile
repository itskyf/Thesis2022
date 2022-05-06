FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER kitti /home/$MAMBA_USER/kitti
