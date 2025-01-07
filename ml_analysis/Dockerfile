FROM continuumio/miniconda3:24.9.2-0 AS ml

WORKDIR /app

RUN conda install -c conda-forge conda-pack

COPY environment.yml environment.yml
RUN conda env create -f environment.yml &&\
    conda-pack -j -1 -n ml_analysis -o /venv_ml --format no-archive &&\
    /venv_ml/bin/conda-unpack


FROM ubuntu:jammy-20240911.1

ENV TMPDIR=/out/
ENV PYTHONDONTWRITEBYTECODE=FALSE

COPY --from=ml /venv_ml/ /venv_ml/

WORKDIR /app

COPY . .

ENTRYPOINT ["/venv_ml/bin/python", "main_fold.py"]