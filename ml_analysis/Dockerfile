FROM continuumio/miniconda3:24.9.2-0 AS ml

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml && conda clean -afy && rm -rf /root/.cache/pip/ && rm -rf /tmp/*

SHELL ["conda", "run", "-n", "ml_analysis", "/bin/bash", "-c"]

RUN mkdir /in && mkdir /out

ENV PYTHONDONTWRITEBYTECODE='false'
#RUN python -m sklearnex.glob patch_sklearn

#RUN awk 'NR < 151 || NR > 156 ' /opt/conda/envs/ml_analysis/lib/python3.12/site-packages/sklearn/utils/deprecation.py >> deprecation.py && \
#    rm -rf /opt/conda/envs/ml_analysis/lib/python3.12/site-packages/sklearn/utils/deprecation.py && \
#    mv deprecation.py /opt/conda/envs/ml_analysis/lib/python3.12/site-packages/sklearn/utils/

COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_analysis", "python", "main_fold.py"]
