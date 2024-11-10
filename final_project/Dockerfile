# 1. Set up based on an Ubuntu image
FROM ubuntu:20.04

# 2. Update and install required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git

# 3. Install required libraries
RUN pip3 install numpy scipy matplotlib

# 4. Setup workiong directory
WORKDIR /bri519-final_project

# 5. Copy source code
COPY . /bri519-final_project

# 6. 
LABEL org.opencontainers.image.source=https://github.com/vlsisp-huiwon/bri519-final_project

# 7. Run main.py
CMD ["python3", "main.py"]

