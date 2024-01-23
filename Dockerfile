FROM python:3.10-bullseye AS builder
COPY requirements.txt /tmp/
RUN pip install --user -r /tmp/requirements.txt

FROM python:3.10-slim-bullseye
WORKDIR /usr/airflow/app
COPY --from=builder /root/.local /root/.local
COPY  __main__.py ./
COPY  src ./src
ENV PATH=/root/.local:$PATH
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
CMD ["python", "."]
