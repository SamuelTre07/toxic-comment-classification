# base docker image
FROM python:3.12.11-slim

# to install uv
RUN pip install --default-timeout=1000 uv

# set working directory
WORKDIR /app

# This "activates" the virtual environment so uvicorn is found automatically
ENV PATH="/app/.venv/bin:$PATH"

# copy the environment dependency files
COPY ".python-version" "pyproject.toml" "uv.lock" "en_core_web_sm-3.8.0-py3-none-any.whl" ./

# installing the dependencies
RUN UV_HTTP_TIMEOUT=500 uv sync --locked

# a tool needed for the text cleaning pipeline in cleaning.py
RUN uv run python -m nltk.downloader stopwords

# to copy the scripts needed for inference, the model file and threshold file
COPY "predict.py" "toxic_comment_prediction_model.pkl" "thresholds.json" "cleaning.py" "contractions.py" ./

# to expose port 8080 from the docker system
EXPOSE 8080

# code to be run on starting up the container
ENTRYPOINT [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080" ]