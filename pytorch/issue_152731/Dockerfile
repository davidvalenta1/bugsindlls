FROM python:3.10

WORKDIR /app
COPY test_issue_152731.py requirements.txt ./

# Remove system pip config that is forcing --index-url
RUN rm -f /etc/pip.conf ~/.pip/pip.conf || true

# Set correct indexes explicitly
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]

