FROM continuumio/miniconda3:24.9.2-0 AS ml

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "ml_analysis", "/bin/bash", "-c"]

RUN mkdir /in && mkdir /out

ENV PYTHONDONTWRITEBYTECODE='false'
RUN python -m sklearnex.glob patch_sklearn

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_analysis", "python", "main_fold.py"]
