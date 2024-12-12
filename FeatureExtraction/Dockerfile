FROM continuumio/miniconda3:24.9.2-0 AS builderPy

WORKDIR /app
COPY external/SATfeatPy/environment.yml satfeatpy.yml

RUN conda install -c conda-forge conda-pack

RUN conda env create -f satfeatpy.yml &&\
    conda-pack -j -1 -n fe_SATfeatPy -o /venv_SATfeatPy --format no-archive &&\
    /venv_SATfeatPy/bin/conda-unpack

FROM gradle:8.11.1-jdk21-jammy AS builderJava

WORKDIR /app
COPY . .
RUN gradle build --no-daemon

FROM eclipse-temurin:21.0.5_11-jre-jammy

RUN mkdir /out/ && mkdir /in/

ENV TMPDIR=/out/

COPY --from=builderPy /venv_SATfeatPy/ /venv_SATfeatPy/

WORKDIR /app

COPY  external/revisiting_satzilla/SAT-features-competition2024/ /app/external/revisiting_satzilla/SAT-features-competition2024/

COPY external/SATfeatPy/feature_computation /app/external/SATfeatPy/feature_computation
COPY external/SATfeatPy/sat_instance /app/external/SATfeatPy/sat_instance
COPY external/SATfeatPy/SatELite/SatELite_v1.0_linux /app/external/SATfeatPy/SatELite/SatELite_v1.0_linux
COPY external/SATfeatPy/ubcsat/ubcsat_linux /app/external/SATfeatPy/ubcsat/ubcsat_linux
COPY external/SATfeatPy/generate_features.py /app/external/SATfeatPy/generate_features.py


COPY --from=builderJava /app/build/libs/FeatureExtraction-1.0-SNAPSHOT-all.jar fe.jar
ENTRYPOINT ["java", "-jar", "fe.jar"]
