# Dirty changes in a branch, return to main without staging committing

If you don't want to commit your changes and you want to switch back to the main
branch, you can use the `git stash` command. The `git stash` command will "
stash" changes, meaning that it takes both staged and unstaged modifications,
saves them for later use, and then reverts them from your working directory.

Here are the steps:

1. Ensure you're on branch B with dirty changes:
    ```
    git checkout B
    ```

2. Stash your changes:
    ```
    git stash
    ```
   This command will save your changes to a new stash which you can reapply
   later.

3. After stashing your changes, you can now switch back to your main branch:
    ```
    git checkout main
    ```

When you want to return to your dirty changes on branch B, switch back to branch
B and apply the stash:

1. Checkout to branch B:
    ```
    git checkout B
    ```

2. Apply the stashed changes:
    ```
    git stash pop
    ```
   This command will apply the changes saved in the stash to your working
   directory and then remove the stash.

If you want to keep the stash after applying it, you can use `git stash apply`
instead of `git stash pop`. The `apply` command leaves the stash in your list of
stashes.

Remember, `git stash` operates on both staged and unstaged changes, so
everything you've modified since your last commit will be stashed.