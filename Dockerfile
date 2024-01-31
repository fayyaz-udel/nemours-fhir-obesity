FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD ["./inference/main.py" ]