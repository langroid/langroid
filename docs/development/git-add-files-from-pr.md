# When main branch has been rebases with squashes/fixups

Trying to merge main into a PR branch can be problematic in this scenario,
if the PR branch has not had the same squashes/fixups applied to it.
However if there are only a few files you need from the PR, you can create a
new branch from the updated main branch and then manually copy over the changes
for those
specific files. Here's how:

1. Checkout the main branch and pull the latest changes:

    ```bash
    git checkout main
    git pull origin main
    ```

2. Create a new branch from main:

    ```bash
    git checkout -b new-branch
    ```

3. Checkout the specific files you need from your PR branch:

    ```bash
    git checkout your-PR-branch path/to/file1 path/to/file2
    ```

   Replace `path/to/file1 path/to/file2` with the paths to the files you want to
   checkout.

4. Now the changes from those files in your PR branch are in your new branch.
   Add and commit these changes:

    ```bash
    git add .
    git commit -m "Copied changes from PR branch"
    ```

5. Push the new branch to GitHub:

    ```bash
    git push origin new-branch
    ```

Now, you should be able to create a new pull request with the `new-branch`. This
method is much simpler, but it only copies over the final state of the specified
files from your PR branch. If there were important changes in other files or
earlier commits, those won't be included.