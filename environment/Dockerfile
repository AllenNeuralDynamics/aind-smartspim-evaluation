# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update -y \
    && apt-get install libgl1 -y \
    && rm -rf /var/lib/apt/lists/*

# Creating a new conda environment for argschema python bug
RUN conda create -n smartspim_eval python=3.9

SHELL ["conda", "run", "-n", "smartspim_eval", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    'xmltodict==0.13.0' \
    'natsort==8.4.0' \
    'numcodecs==0.12.1' \
    'matplotlib==3.7.2' \
    'numpy==1.25.2' \
    'argschema==3.0.4' \
    'scikit-image==0.22.0' \
    'zarr==2.16.1' \
    'ome-zarr==0.8.3' \
    'matplotlib==3.8.2' \
    'pims==0.6.1' \
    'tifffile==2023.8.25' \
    'phasepack==1.5' \
    'pyfftw==0.13.1' \
    'natsort==8.4.0' \
    'dask[distributed]==2022.11.1' \
    'opencv-python==4.8.0.76' \
    'dask-image==2023.3.0'