FROM --platform=$TARGETPLATFORM python:3.11

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install necessary tools, zsh, and set up locale
RUN apt-get update && \
    apt-get install --no-install-recommends -y zsh wget git curl locales \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libssl-dev && \
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

RUN mkdir -p /root/.cache/uv

# workaround for pymupdf build error?
ENV MAKEFLAGS="-j1"
ENV PYTHONPYCACHEPREFIX="/tmp/pycache"
ENV DEBIAN_FRONTEND=non-interactive \
     LANG=en_US.UTF-8

# detect arch to customize pymupdf version
ARG TARGETPLATFORM
ARG TARGETARCH

# install uv then langroid
# Install uv and use it with cache mount
RUN --mount=type=cache,target=/root/.cache/uv,id=uv_cache \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    if [ "$TARGETARCH" = "arm64" ]; then \
         uv pip install --no-cache-dir "pymupdf==1.24.14"; \
     else \
         uv pip install --no-cache-dir "pymupdf>=1.25.3"; \
     fi && \
    uv pip install --no-cache-dir .

# Install oh-my-zsh and set up zsh configurations
RUN sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)" || true && \
    sed -i -e 's/plugins=(git)/plugins=(git python)/' /root/.zshrc

CMD ["zsh"]