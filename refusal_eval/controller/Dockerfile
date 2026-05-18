FROM python:3.12-slim

# docker.io: docker CLI + Python SDK base for talking to the host daemon
# via the mounted /var/run/docker.sock.
# docker-compose (v1 from debian): compose CLI; we use `-f`/`-p` flags only,
# which are stable across v1/v2. Avoids adding Docker's apt repo.
# curl: used by the compose healthcheck against /healthz.
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io docker-compose curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir aiohttp==3.10.10 docker==7.1.0

WORKDIR /app
COPY refusal_eval/controller/controller.py /app/controller.py

# RPC port. Stays clear of the grader's reserved 9999/9998/9997/9996.
EXPOSE 9990

CMD ["python3", "/app/controller.py"]
