FROM quay.io/astronomer/astro-runtime:11.7.0-python-3.10

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER astro

RUN pip3 install -r requirements.txt