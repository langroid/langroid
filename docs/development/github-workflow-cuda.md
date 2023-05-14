To install CUDA 11.7 in a GitHub Actions workflow, you need to create or update
your `.github/workflows/main.yml` file with the necessary steps. Here's an
example of a basic workflow configuration that installs CUDA 11.7 on an
Ubuntu-based runner:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake

      - name: Install CUDA Toolkit 11.7
        run: |
          wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
          sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
          sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
          sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
          sudo apt-get update
          sudo apt-get -y install cuda-11-7

      - name: Verify CUDA installation
        run: |
          nvcc --version

      - name: Build and test
        run: |
          # Your build and test commands here
```

This workflow configuration performs the following steps:

1. Triggers the workflow on pushes and pull requests to the `main` branch.
2. Runs on the latest Ubuntu runner.
3. Checks out your repository.
4. Installs build dependencies.
5. Downloads and installs the CUDA Toolkit 11.7.
6. Verifies the CUDA installation by checking the `nvcc` version.
7. Builds and tests your project (replace the comment with your actual build and
   test commands).

Make sure to replace the build and test commands in the last step with the
commands specific to your project.

# Using a Docker Image

If you're looking for a more straightforward way to set up CUDA 11.7 in a GitHub
Actions workflow, you can use a pre-built Docker image with CUDA installed.
Here's an example of a basic workflow configuration using an NVIDIA CUDA Docker
image:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    container:
      image: nvidia/cuda:11.7.0-devel-ubuntu20.04

    steps:
      - name: Checkout repository
        uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y build-essential cmake

      - name: Verify CUDA installation
        run: |
          nvcc --version

      - name: Build and test
        run: |
          # Your build and test commands here
```

This workflow configuration:

1. Triggers the workflow on pushes and pull requests to the `main` branch.
2. Runs on the latest Ubuntu runner.
3. Uses the NVIDIA CUDA 11.7.0 Docker image based on Ubuntu 20.04 as a
   container.
4. Checks out your repository.
5. Installs build dependencies.
6. Verifies the CUDA installation by checking the `nvcc` version.
7. Builds and tests your project (replace the comment with your actual build and
   test commands).

This approach is simpler because it uses a pre-built Docker image containing the
desired version of CUDA, so you don't have to manually install it in the
workflow. Make sure to replace the build and test commands in the last step with
the commands specific to your project.