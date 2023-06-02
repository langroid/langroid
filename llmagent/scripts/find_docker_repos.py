import csv
import os

import fire
from dotenv import load_dotenv
from github import Github


def find_docker_repos(stars: int = 10000, k: int = 10, out: str = None):
    load_dotenv()
    access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    g = Github(access_token)

    # Define the search query
    query = f"language:python stars:>={stars}"

    # Fetch the repositories
    repositories = g.search_repositories(query=query)

    # Filter repositories with Dockerfile in the root directory
    filtered_repos = []

    for repo in repositories:
        if len(filtered_repos) >= k:
            break

        try:
            dockerfile = repo.get_contents("Dockerfile")
            readme = repo.get_contents("README.md")
            if dockerfile and dockerfile.type == "file":
                if readme and readme.type == "file":
                    # restrict to repos where the readme mentions docker
                    if "docker run" in readme.decoded_content.decode().lower():
                        filtered_repos.append(repo)
        except Exception:
            pass

    # Print the filtered repository names and URLs
    c = 0
    save_repos_to_csv = []
    for repo in filtered_repos:
        print(f"{c}, {repo.name}, {repo.html_url}, {repo.stargazers_count}")
        row = [c, repo.name, repo.html_url, repo.stargazers_count]
        save_repos_to_csv.append(row)
        c += 1

    if out is not None:
        with open(out, "w") as csvfile:
            csvwriter = csv.writer(csvfile)

            # writing the header
            csvwriter.writerow(["No.", "Repo_name", "URL", "Stars"])

            # writing each repo as a separate row in the csv
            for repo in save_repos_to_csv:
                csvwriter.writerow(repo)


if __name__ == "__main__":
    fire.Fire(find_docker_repos)
