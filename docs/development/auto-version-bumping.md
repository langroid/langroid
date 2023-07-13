Sure, you can do this with a combination of git hooks and a version-bumping
tool. A git hook is a script that Git executes before or after events such
as: `commit`, `push`, and `receive`. The hooks are all stored in
the `.git/hooks` directory of every Git repository.

Here is a guide on how you might set up automatic version bumping using a tool
called `bumpversion`, which is a Python utility to automate the incrementing of
semantic versioning numbers:

1. First, install bumpversion via pip:

    ```
    pip install --upgrade bumpversion
    ```

2. Create a `.bumpversion.cfg` file in your project's root directory. This file
   will define how `bumpversion` operates.

   Here is an example configuration for a simple project:

    ```cfg
    [bumpversion]
    current_version = 0.0.1
    commit = True
    tag = True
    message = "Bump version: {current_version} → {new_version}"

    [bumpversion:file:setup.py]
    ```

3. Install `bumpversion` as a pre-push git hook. You can do this by creating a
   file named `pre-push` in your `.git/hooks` directory.

   Here is an example `pre-push` hook that bumps the version:

    ```bash
    #!/bin/sh
    bumpversion patch
    ```

   Ensure the `pre-push` hook is executable:

    ```bash
    chmod +x .git/hooks/pre-push
    ```

   This will bump the "patch" version every time you push. If you want to bump
   the "major" or "minor" version, you can just run `bumpversion major`
   or `bumpversion minor` manually.

Please note that this example is very simple, and it doesn't consider the branch
that you are pushing to. If you only want to bump the version when pushing to
the main branch, you can modify the `pre-push` hook script to check the current
branch:

```bash
#!/bin/sh
branch="$(git rev-parse --abbrev-ref HEAD)"

if [ "$branch" = "main" ]; then
  bumpversion patch
fi
```

This version of the script will only bump the version if the current branch is "
main".

Bear in mind that, depending on your project's needs, you may want to adjust
this process. For instance, you might want to only bump the version when merging
a pull request, or you may want to bump different parts of the version under
different conditions. You can do this by adjusting your `.bumpversion.cfg` and
your git hooks accordingly.