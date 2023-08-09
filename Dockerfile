FROM python:3.11

# Install git
RUN apt-get update && apt-get install -y git

# Clone the langroid repository
RUN git clone https://github.com/langroid/langroid.git

# Set the working directory in the container
WORKDIR /langroid

RUN mv .env-template .env

# Install the langroid package via pip
RUN pip install langroid

CMD ["tail", "-f", "/dev/null"]