FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
COPY . /app

ENTRYPOINT [ "python","./inference/main.py" ]
CMD [ "python3","-m", "http.server" ,"80"]

