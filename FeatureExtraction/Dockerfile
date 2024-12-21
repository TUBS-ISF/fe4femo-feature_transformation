FROM continuumio/miniconda3:24.9.2-0 AS satfeatpy

WORKDIR /app

RUN conda install -c conda-forge conda-pack

COPY external/SATfeatPy/environment.yml satfeatpy.yml
RUN conda env create -f satfeatpy.yml &&\
    conda-pack -j -1 -n fe_SATfeatPy -o /venv_SATfeatPy --format no-archive &&\
    /venv_SATfeatPy/bin/conda-unpack

FROM continuumio/miniconda3:24.9.2-0 AS fmchara

WORKDIR /app

RUN conda install -c conda-forge conda-pack

COPY external/fm_characterization/environment.yml fmchara.yml
RUN conda env create -f fmchara.yml &&\
    conda-pack -j -1 -n fm_characterization -o /venv_fm_chara --format no-archive &&\
    /venv_fm_chara/bin/conda-unpack

FROM gradle:8.11.1-jdk21-jammy AS java

WORKDIR /app
COPY . .
RUN gradle build --no-daemon

FROM ubuntu:jammy-20240911.1

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-21-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /out/ && mkdir /in/

ENV TMPDIR=/out/

COPY --from=satfeatpy /venv_SATfeatPy/ /venv_SATfeatPy/
COPY --from=fmchara /venv_fm_chara/ /venv_fm_chara/

WORKDIR /app

COPY  external/revisiting_satzilla/SAT-features-competition2024/ /app/external/revisiting_satzilla/SAT-features-competition2024/

COPY external/feature-model-batch-analysis/solver/ /app/external/feature-model-batch-analysis/solver/

COPY external/SATfeatPy/feature_computation /app/external/SATfeatPy/feature_computation
COPY external/SATfeatPy/sat_instance /app/external/SATfeatPy/sat_instance
COPY external/SATfeatPy/SatELite/SatELite_v1.0_linux /app/external/SATfeatPy/SatELite/SatELite_v1.0_linux
COPY external/SATfeatPy/ubcsat/ubcsat_linux /app/external/SATfeatPy/ubcsat/ubcsat_linux
COPY external/SATfeatPy/generate_features.py /app/external/SATfeatPy/generate_features.py

COPY external/fm_characterization/fm_characterization /app/external/fm_characterization/fm_characterization
COPY external/fm_characterization/main_characterization.py /app/external/fm_characterization/main_characterization.py


COPY --from=java /app/build/libs/FeatureExtraction-1.0-SNAPSHOT-all.jar fe.jar
ENTRYPOINT ["java", "-jar", "-Djava.io.tmpdir=/out/", "fe.jar"]
