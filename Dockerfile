FROM python:3.11

# Set the working directory in the container
WORKDIR /langroid

# Install git
RUN apt-get update && apt-get install -y git

# Clone the langroid-examples repository
RUN git clone https://github.com/langroid/langroid-examples.git

# Copy the .env-template file from the GitHub repository
ADD https://raw.githubusercontent.com/langroid/langroid/main/.env-template .env

# Install the langroid package via pip
RUN pip install langroid

CMD ["tail", "-f", "/dev/null"]