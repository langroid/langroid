FROM python:3.11

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install necessary tools and zsh
RUN apt-get update && apt-get install -y \
    zsh \
    wget \
    git \
    curl

RUN apt-get install -y locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

# Set the locale environment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Clone the langroid repository
RUN git clone https://github.com/langroid/langroid.git

# Set the working directory in the container
WORKDIR /langroid

RUN mv .env-template .env

# Install the langroid package via pip
RUN pip install langroid

# Install oh-my-zsh
RUN sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)" || true

# Set theme to "agnoster" and enable some plugins
RUN sed -i -e 's/plugins=(git)/plugins=(git python)/' /root/.zshrc

CMD ["zsh"]