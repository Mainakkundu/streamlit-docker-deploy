FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

#TMPDIR  /var/tmp/ pip install --cache-dir=/data/vincents/ --build /data/vincents/ tensorflow-gpu

RUN pip3 install -r requirements.txt
## as docker error not able to install so I just put it into a seperate command 
RUN pip3 install --no-cache-dir tfcausalimpact=0.0.6

EXPOSE 8080 

COPY . /app

#ENTRYPOINT [ "streamlit","run" ]

CMD streamlit run --server.port 8080 --server.enableCORS false app.py