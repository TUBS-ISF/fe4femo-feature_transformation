FROM continuumio/miniconda3:24.9.2-0 AS ml

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "ml_analysis", "/bin/bash", "-c"]

RUN mkdir /in && mkdir /out

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_analysis", "python", "main_fold.py"]
