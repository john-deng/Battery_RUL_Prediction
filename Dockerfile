FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
LABEL maintainer="<john.deng@outlook.com>"

RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
    -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
    /etc/apt/sources.list

# 1) Install python3, venv support and pip via apt
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip \
 && rm -rf /var/lib/apt/lists/*

# 2) Create a real venv with its own pip
RUN python3 -m venv /opt/venv

# 3) Make the venv the default Python environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4) (Optional) point pip at a faster mirror & bump timeouts
ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=1

# 5) Upgrade pip & setuptools inside the venv
RUN pip install --upgrade pip setuptools

# set working directory
WORKDIR /app

# copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# copy your training code
COPY . .

# mount your datasets at runtime under /app/Datasets
VOLUME ["/app/Datasets"]

# default command: run your training script
# adjust "train.py" if your script has a different filename
CMD ["python3", "train.py"]
