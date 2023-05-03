import subprocess
import requests


def clone_repo(repo_url: str, repo_clone_path):
    repo_clone_result = subprocess.run(["git", "clone", repo_url, repo_clone_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return repo_clone_result.returncode


def extract_repo_metadata(repo_url: str):
    owner, repo_name = repo_url.split("/")[-2:]
    repo_metadata = {}
    # Construct the URL of the GitHub API endpoint
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"

    # Send a GET request to the GitHub API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
        
        # Extract the language and dependencies
        repo_metadata['language'] = data["language"]
        if repo_metadata['language'] == "Python":
            requirements_url = data["contents_url"].replace("{+path}", "requirements.txt")
            requirements_response = requests.get(requirements_url)
            if requirements_response.status_code == 200:
                repo_metadata['requirements'] = 1
            else:
                repo_metadata['requirements'] = 0
    else:
        print(f"Error retrieving data from GitHub API: {response.status_code}")

    return response.status_code, repo_metadata
