To checkout a pull request (PR) using the GitHub CLI (`gh`), follow these steps:

1. First, ensure you have the GitHub CLI installed. If not,
   visit https://github.com/cli/cli#installation for instructions on how to
   install it.

2. Open your terminal or command prompt and navigate to the local repository
   where you want to checkout the PR.

3. Run the following command to authenticate with your GitHub account (if you
   haven't already):

```
gh auth login
```

4. Follow the prompts to complete the authentication process.

5. To checkout a specific PR, use the following command:

```
gh pr checkout <pr-number>
```

Replace `<pr-number>` with the number of the pull request you want to checkout.

This command will create a new branch in your local repository, fetch the
changes from the PR, and switch your working tree to that branch. Now you can
review, test, and make additional changes to the PR as needed.

Remember to replace `<pr-number>` with the actual number of the pull request you
want to checkout.

# Viewing the PR changes in PyCharm

Yes, you can see the changes relative to the main branch in PyCharm by following
these steps:

1. Checkout the pull request (PR) as previously described using the GitHub CLI.
   This will create a new branch and fetch the changes from the PR.

2. Open your project in PyCharm.

3. If you have not added the repository as a Git remote, you can do this by
   going to `VCS` > `Git` > `Remotes...` in the PyCharm menu. Add the remote
   repository URL if it's not already listed.

4. In the bottom-right corner of the PyCharm window, click on the Git branch
   indicator (it should show the current branch you are on, which is the PR
   branch).

5. In the pop-up menu, click on the `main` branch (or the branch you want to
   compare the PR to) and select `Compare with Current`.

6. A new tab will open in PyCharm, showing the differences between the PR branch
   and the main branch. In the editor, you will see the changed lines with
   indicators in the gutter.

7. To view the changes in a side-by-side comparison, you can click on any file
   in the differences tab, and PyCharm will open the diff viewer.

With these steps, you can review the changes made in the PR relative to the main
branch and see the modified lines with indicators in the gutter in PyCharm.

# Viewing the PR changes relative to main as "dirty" files

Yes, you can create a new branch, merge the PR branch into it, and then view the
changes as "dirty." Here's how to do that:

1. Checkout the main branch using the following command in your terminal or
   command prompt:

```
git checkout main
```

2. Create a new branch called "pr-view" and switch to it:

```
git checkout -b pr-view
```

3. Merge the PR branch into the "pr-view" branch without committing, using
   the `--no-commit` and `--no-ff` options:

```
git merge --no-commit --no-ff <pr-branch-name>
```

Replace `<pr-branch-name>` with the name of the branch corresponding to the PR.

4. Open the project in PyCharm. Now, all the changes from the PR branch will be
   shown as "dirty" changes relative to the "pr-view" branch. You can see the
   changes with gutter indicators in the editor, just like when you make changes
   to your files.

5. After reviewing the changes, you can switch back to the main branch and
   delete the "pr-view" branch:

```
git checkout main
git branch -D pr-view
```

This approach keeps your main branch clean and lets you view the changes from
the PR branch as "dirty" changes in a separate branch.


