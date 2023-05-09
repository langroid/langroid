You can test GitHub Actions workflows locally using a tool
called `act`. `act` allows you to run your GitHub Actions workflows on your
local machine without pushing your changes to the repository. You can find more
information about the tool on its GitHub
repository: https://github.com/nektos/act

To get started with `act`, follow these steps:

1. Install `act`: Depending on your operating system, you can install `act`
   using one of the following methods:

    - For macOS: `brew install act`
    - For Linux: Download the latest release
      from https://github.com/nektos/act/releases and follow the installation
      instructions.
    - For Windows: Download the latest release
      from https://github.com/nektos/act/releases and follow the installation
      instructions.

2. Run your workflow locally: Once you have `act` installed, open a terminal,
   navigate to your repository directory, and run the following command:

   ```
   act
   ```

   This command will read the GitHub Actions workflow files in
   your `.github/workflows` directory and execute the workflows using Docker
   containers. By default, `act` uses the `nektos/act-environments-ubuntu:18.04`
   Docker image, which closely mimics the GitHub-hosted Ubuntu runners.

3. Customize the local environment: If necessary, you can customize the Docker
   image and environment variables used by `act` to better match the GitHub
   Actions environment. You can find more information about customization in
   the `act`
   documentation: https://github.com/nektos/act#customizing-the-environment

Please note that `act` may not support all GitHub Actions features, and there
might be differences between the local environment and the actual GitHub Actions
environment. However, it is a helpful tool for quickly iterating and testing
workflows before pushing changes to your repository.