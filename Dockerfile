FROM ubuntu:20.04

COPY . /home/ubuntu/python-apis/ai_jira_assistant

WORKDIR /home/ubuntu/python-apis/ai_jira_assistant

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get -y install python3-pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get -y install wget
RUN apt-get install -y chromium-browser

RUN mkdir -p /tmp/html

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
