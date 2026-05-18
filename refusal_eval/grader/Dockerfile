FROM python:3.12-slim

# Slim DB drivers for db_row_added / db_table_added probes.
# Web probes are mostly stdlib, but Flask-session auth proofs need
# itsdangerous + Flask's TaggedJSONSerializer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 docker.io \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir psycopg2-binary==2.9.9 PyMySQL==1.1.1 docker==7.1.0 itsdangerous==2.2.0 Flask==3.1.0

WORKDIR /app

# The grader module tree is mounted in or copied from the repo at build time.
# In the compose, the repo is mounted read-only at /app/refusal_eval so that
# iterating on the probe code does not require rebuilding the image.
ENV PYTHONPATH=/app

COPY refusal_eval/grader /app/refusal_eval/grader
RUN touch /app/refusal_eval/__init__.py

# Primary auth'd endpoint for /baseline and /verify.
EXPOSE 9999

# Unauthenticated inbound listener for outbound_webhook probe callbacks.
# Only reachable from inside shared_net; never mapped to host ports.
EXPOSE 9998

# Raw TCP listeners for jndi_listener probe: LDAP (9997), RMI (9996).
# Both are shared_net-only (not host-mapped). They don't emulate the
# protocols — the probe only verifies the JNDI lookup fired, not RCE.
EXPOSE 9997
EXPOSE 9996

CMD ["python3", "-m", "refusal_eval.grader.grader"]
