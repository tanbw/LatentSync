FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 设置环境变量
ARG VENV_NAME="latentsync"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing && apt-get install -y --no-install-recommends \
    wget \
    git \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://ghfast.top/github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH /opt/conda/bin:$PATH
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

RUN conda create -y -n ${VENV} python=3.10.13
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace
RUN git clone https://ghfast.top/github.com/tanbw/LatentSync.git && \
    cd LatentSync && \
    git submodule update --init --recursive

WORKDIR /workspace/LatentSync
RUN conda activate ${VENV} && conda install -y -c conda-forge ffmpeg
RUN conda activate ${VENV}  && \
 pip install -r requirements.txt && pip install huggingface_hub[hf_xet]

# 暴露 Gradio 默认端口
EXPOSE 7860
COPY ./api.py /workspace/LatentSync/api.py
# 启动命令：激活 Conda 环境后运行 Gradio 应用
CMD ["/bin/bash", "-c", "source ~/.bashrc && python api.py"]