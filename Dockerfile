FROM python:3.11

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install necessary tools, zsh, and set up locale
RUN apt-get update && \
    apt-get install --no-install-recommends -y zsh wget git curl locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    # Cleanup apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the langroid repository
RUN git clone https://github.com/langroid/langroid.git

# Set the working directory in the container
WORKDIR /langroid
RUN mv .env-template .env

# Install the langroid package via pip
RUN pip install --no-cache-dir langroid

# Install oh-my-zsh and set up zsh configurations
RUN sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)" || true && \
    sed -i -e 's/plugins=(git)/plugins=(git python)/' /root/.zshrc

CMD ["zsh"]