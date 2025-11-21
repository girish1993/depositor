FROM python:3.11-slim


WORKDIR /app

# Copy code & configs
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt

# Copy code & configs
COPY src ./src
COPY configs ./configs


CMD ["python", "-m", "src.modeling.trainer"]
