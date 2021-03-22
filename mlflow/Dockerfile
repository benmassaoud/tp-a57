FROM python:3.8.0

LABEL maintainer "Zakaria Benmassaoud <benmassaoud@gmail.com>"

RUN mkdir /mlflow/

RUN pip install mlflow

EXPOSE 5000

CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts  --host 0.0.0.0