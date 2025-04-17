FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/envs/webapp/bin:$PATH"

# 安裝工具（如 netcat）
RUN apt-get update && apt-get install -y netcat-openbsd

# 建立非 root 使用者
RUN useradd -m -s /bin/bash appuser

# ARG → 傳進來再設定為 ENV
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

# 複製並設定 entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser
WORKDIR /home/appuser

# 安裝 Conda 環境
COPY --chown=appuser:appuser ./environment.yml /home/appuser/environment.yml
RUN conda env create --name webapp -f environment.yml && conda clean -afy

# 複製 Flask 專案
WORKDIR /home/appuser/flask_backend
COPY --chown=appuser:appuser . /home/appuser/flask_backend

EXPOSE ${BACKEND_PORT}

ENTRYPOINT ["/entrypoint.sh"]
