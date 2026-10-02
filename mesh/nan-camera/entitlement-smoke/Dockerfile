FROM python:3.12-alpine
COPY smoke.py /usr/local/bin/nan-entitlement-smoke
USER 1000:1000
ENTRYPOINT ["python3", "/usr/local/bin/nan-entitlement-smoke"]
