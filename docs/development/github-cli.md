# Using the GitHub CLI

!!! note
    By GPT4. Caveat Lector. May not be fully accurate. Trust but Verify!


`gh` is the GitHub CLI (Command Line Interface) tool that helps you interact
with GitHub directly from your command line. It enhances your Git workflow by
allowing you to manage issues, pull requests, repositories, and more without
leaving the terminal. To get started, you need to install `gh` first.
Visit https://github.com/cli/cli#installation for instructions on how to install
it on your system.

Here are some basics for using `gh` when working with Git:

1. Authenticate with GitHub:

   ```
   gh auth login
   ```

   Follow the prompts to log in to your GitHub account.

2. Create a new repository:

   ```
   gh repo create <repo_name>
   ```

   Replace `<repo_name>` with the desired repository name. This will create a
   new repository on GitHub and add the remote origin to your local Git
   repository.

3. Clone a repository:

   ```
   gh repo clone <owner>/<repo_name>
   ```

   Replace `<owner>` with the repository owner's username and `<repo_name>` with
   the repository name. This command clones the specified GitHub repository to
   your local machine.

4. Create a new issue:

   ```
   gh issue create --title "<issue_title>" --body "<issue_description>"
   ```

   Replace `<issue_title>` with the title of the issue and `<issue_description>`
   with a brief description of the issue. This command creates a new issue in
   the current repository.

5. List issues:

   ```
   gh issue list
   ```

   This command shows a list of open issues in the current repository.

6. Create a new pull request:

   ```
   gh pr create --title "<pr_title>" --body "<pr_description>"
   ```

   Replace `<pr_title>` with the title of the pull request
   and `<pr_description>` with a brief description of the pull request. This
   command creates a new pull request for the current branch.

7. List pull requests:

   ```
   gh pr list
   ```

   This command shows a list of open pull requests in the current repository.

8. View, comment, and manage pull requests:

   ```
   gh pr view <pr_number>
   gh pr review <pr_number>
   gh pr merge <pr_number>
   ```

   Replace `<pr_number>` with the pull request number. These commands let you
   view, comment, and merge pull requests.

These are just some of the basic `gh` commands. You can find a comprehensive
list of commands and their descriptions in the official GitHub CLI
documentation: https://cli.github.com/manual/