FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安裝必要工具與 Miniconda
RUN apt-get update && apt-get install -y wget bzip2 git netcat-openbsd \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_DEFAULT_ENV=webapp

# 建立非 root 使用者
RUN useradd -m -s /bin/bash appuser

ARG OPENAI_APIKEY
ARG BACKEND_PORT
ARG BACKEND_API_URL_TEXT_GENERATE
ARG BACKEND_API_URL_TEXT_REPHRASE
ARG BACKEND_API_URL_PDF_GENERATE
ARG BACKEND_API_URL_PDF_REPHRASE

ENV OPENAI_APIKEY=${OPENAI_APIKEY}
ENV BACKEND_PORT=${BACKEND_PORT}
ENV BACKEND_API_URL_TEXT_GENERATE=${BACKEND_API_URL_TEXT_GENERATE}
ENV BACKEND_API_URL_TEXT_REPHRASE=${BACKEND_API_URL_TEXT_REPHRASE}
ENV BACKEND_API_URL_PDF_GENERATE=${BACKEND_API_URL_PDF_GENERATE}
ENV BACKEND_API_URL_PDF_REPHRASE=${BACKEND_API_URL_PDF_REPHRASE}

# 設定 Conda 環境
COPY --chown=appuser:appuser ./environment.yml /home/appuser/environment.yml
RUN conda env create -n webapp -f /home/appuser/environment.yml && conda clean -afy
ENV PATH="/opt/conda/envs/webapp/bin:$PATH"

# 複製 app 並設定權限
WORKDIR /home/appuser/flask_backend
COPY --chown=appuser:appuser . /home/appuser/flask_backend

RUN mkdir -p /app/uploaded_pdfs && chmod 777 /app/uploaded_pdfs

# 複製 entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# USER appuser

EXPOSE ${BACKEND_PORT}
ENTRYPOINT ["/entrypoint.sh"]
