# Import python for running code
FROM python:latest

# Settitng working directory to f.e. /app
WORKDIR ...

COPY . requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /app

# command for running application
CMD []