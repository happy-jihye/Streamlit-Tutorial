FROM python:3

CMD mkdir /app
COPY . /app

WORKDIR /app

EXPOSE 8502

RUN pip3 install -r requirements.txt

CMD streamlit run app.py
