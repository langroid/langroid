# Searching on GitHub


!!! note
    Caveat Lector. May not be fully accurate. Trust but Verify!

Unfortunately, GitHub's search syntax
does not provide a direct way to specify the root level path for files in the
search query. However, we can still use an alternative approach by manually
scanning the search results.

**Manually:**

1. Go to https://github.com/
2. Use the search bar at the top and type the following query:

```
language:python stars:>=100 Dockerfile in:file
```

This query searches for Python repositories with at least 100 stars containing
the file named 'Dockerfile'.

After finding a repository, you will need to manually verify if the Dockerfile
is in the root directory by opening the repository and checking the root
directory for the presence of a Dockerfile.

**Programmatically using PyGitHub:**

The previous programmatic example should work correctly for your requirements,
as it filters the results to show only repositories with Dockerfiles in the root
directory. Here it is again for reference:

```python
from github import Github

# Replace with your personal access token or use an empty string for unauthenticated requests (limited rate)
access_token = 'your_personal_access_token'
g = Github(access_token)

# Define the search query
query = 'language:python stars:>=100 Dockerfile in:file'

# Fetch the repositories
repositories = g.search_repositories(query=query)

# Filter repositories with Dockerfile in the root directory
filtered_repos = []

for repo in repositories:
    contents = repo.get_contents("/")
    dockerfile_found = False

    for content in contents:
        if content.type == "file" and content.name == "Dockerfile":
            dockerfile_found = True
            break

    if dockerfile_found:
        filtered_repos.append(repo)

# Print the filtered repository names and URLs
for repo in filtered_repos:
    print(f"{repo.name}: {repo.html_url}")
```

Replace `'your_personal_access_token'` with your GitHub personal access token.
If you don't have one, you can
follow [these instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
to create one.