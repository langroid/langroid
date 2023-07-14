Great! Given that your poetry environment is fully set up, and your code is
ready to go, here's how to build, publish, and install a Python package from a
private GitHub repository.

# Building a package using Poetry and publishing it to GitHub

1. **Building your package**

   You can build your package using the `poetry build` command:

   ```bash
   poetry build
   ```

   This will create a `.tar.gz` file and a `.whl` file in the `dist` directory.

2. **Publishing your package**

   For a private GitHub repository, you typically do not publish your package to
   PyPI or other package repositories. Instead, you create a release with your
   built distributions.

   You can do this by going to your GitHub repository, then clicking on "
   Releases", then "Draft a new release". Upload your `.tar.gz` and `.whl` files
   as release assets. Then, give your release a tag (usually, this is the
   version number of your package), and a title and description.

3. **Using the package**

   You can then install your package directly from the GitHub release
   using `pip`.

   Here's how you can do it:

   ```bash
   pip install "git+https://<token>:x-oauth-basic@github.com/<username>/<repo>@<tag>#egg=<package_name>"
   ```

   Replace `<token>` with your GitHub personal access token, `<username>` with
   your GitHub username, `<repo>` with the repository name, `<tag>` with the tag
   of your release (usually the version number), and `<package_name>` with the
   name of your package.

   You can generate a personal access token by going to your GitHub settings,
   then "Developer settings", then "Personal access tokens".

Note: Directly embedding credentials into a pip install command is generally not
a good practice, especially in a shared or production environment, due to
security concerns. The method described is a way to install directly from a
private repository for personal usage or for quick tests.

In production, consider using environment variables, GitHub Apps, or other
secure methods to handle authentication. It's also recommended to consider using
a private PyPI server for hosting Python packages if you regularly use private
packages.

# Version bumping

Yes, using version bumping utilities is definitely a good idea. Version bumping
tools such as `bumpversion` or `bump2version` help automate the process of
updating your project's version number. This is very important in maintaining
proper software versioning standards.

Here's how you could potentially use `bump2version` in this context:

1. **Installation**

   First, you need to install the `bump2version` tool. You can install it using
   pip:

    ```bash
    pip install --upgrade bump2version
    ```

2. **Configuration**

   `bump2version` works by modifying specific files in your project where the
   current version number is stored. So, you need to tell `bump2version` where
   it can find the version number in your project.

   You do this by creating a `.bumpversion.cfg` file in your project's root
   directory. Here's an example `.bumpversion.cfg` for a Python project that
   uses `poetry` (this file should be tracked in git,
   so you should do a `git add .bumpversion.cfg`):

     ```ini
    .bumpversion.cfg`):

    ```ini
    [bumpversion]
    current_version = 0.1.0
    commit = True
    tag = True

    [bumpversion:file:pyproject.toml]
    ```

   In this configuration file, `current_version` is your project's current
   version number. `commit = True` tells `bump2version` to create a commit
   whenever the version number is updated. `tag = True` tells `bump2version` to
   create a tag with the new version number.

   The `[bumpversion:file:pyproject.toml]` part tells `bump2version` to also
   update the version number in your `pyproject.toml` file.

3. **Bumping the version**

   You can now bump your project's version number using the `bumpversion`
   command:

    ```bash
    bump2version patch  # for a patch version increment (e.g., 0.1.0 to 0.1.1)
    bump2version minor  # for a minor version increment (e.g., 0.1.0 to 0.2.0)
    bump2version major  # for a major version increment (e.g., 0.1.0 to 1.0.0)
    ```

4. **Publishing your package**

   You can now build and publish your package. After building
   with `poetry build`, you can create a new GitHub release with the updated
   version number.

By using a tool like `bump2version`, you ensure that your versioning is
consistent and aligned with the best practices of semantic versioning. This is
particularly important for any users or applications that depend on specific
versions of your package.

# Publishing using `gh` cli tool

**Using the Command Line:** You can use the GitHub CLI or git commands to
upload files. Here's how you can do it using GitHub CLI:

Install GitHub CLI, if not already installed:

   ```bash
   brew install gh
   ```

Then, you can create a release and upload a file like this:

   ```bash
   gh release create <tag> dist/* --title "Release title" --notes "Release notes"
   ```

Replace `<tag>` with your release tag, and `dist/*` with the path to
your `.whl` file. This will create a new release and upload your file.

If you're still having trouble, it might be best to contact GitHub support for
help. They may have more specific advice based on the details of your account
and repository.

# Using Poetry to install from the private repo

Yes, Poetry can use environment variables when installing dependencies from a
private repository, but unlike in pip, the token can't be directly used in the
repository URL in the `pyproject.toml` file due to the TOML syntax.

However, there is a way around it by creating a `config.toml` file for Poetry
where you define the private repository with the GitHub token.

Here are the steps to achieve that:

1. Add the private repository to your Poetry configuration:

   You'll want to create a `config.toml` file and place it under the poetry's
   configuration directory. For Unix OS, it's usually
   under `~/.config/pypoetry/`.

    ```toml
    [[tool.poetry.source]]
    name = "private-repo"
    url = "https://<TOKEN>:x-oauth-basic@github.com/username/repo"
    default = true
    ```

   Replace `<TOKEN>` with your GitHub token.

   The `default` flag is set to `true` to prioritize this source over PyPI.

2. In your project's `pyproject.toml` file, specify your package dependencies as
   usual, but also specify the source using the `source` attribute:

    ```toml
    [tool.poetry.dependencies]
    your-private-package = { version = "*", source = "private-repo" }
    ```

   Replace `your-private-package` with the name of your private package.

Remember, storing your token in a configuration file can expose it if the file
is accidentally shared or committed. To avoid this, you could load the token
from an environment variable in a Python script that modifies the Poetry
configuration when needed.

However, this approach may require more complex scripting and careful handling
to avoid accidentally committing a modified configuration file with your token.

A more secure alternative for a production setting would be to use a private
package index server, or a commercial service like GitHub Packages or GitLab's
package repositories. These services can authenticate with deploy keys or
instance-wide tokens that don't need to be embedded in project files.