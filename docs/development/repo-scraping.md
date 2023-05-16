# Authenticated Requests using PyGithub

Here is how to use `pygithub` to get contents of a git repo, without 
hitting github API rate limits.


To perform authenticated requests using PyGithub, you need to provide an access
token or your username and password when creating the `Github` object. Using an
access token is recommended, as it's more secure than using your username and
password.

To create an access token, follow these steps:

1. Go to your GitHub settings: https://github.com/settings/profile
2. In the left sidebar, click on "Developer settings."
3. In the left sidebar, click on "Personal access tokens."
4. Click on the "Generate new token" button.
5. Give your token a description, select the necessary scopes for your
   application (e.g., "repo" for accessing private repositories), and click "
   Generate token."
6. Copy the generated token. Make sure to store it securely, as you won't be
   able to see it again.

Now that you have an access token, you can use it with PyGithub:

```python
from github import Github

# Replace YOUR_ACCESS_TOKEN with the actual token you generated
access_token = "YOUR_ACCESS_TOKEN"
g = Github(access_token)

# Now you can make authenticated requests
repo = g.get_repo("username/reponame")
contents = repo.get_contents("")
```

Remember to replace `"YOUR_ACCESS_TOKEN"` with the actual token you generated.

Keep in mind that using an access token will significantly increase your API
rate limit, but there are still limits. For authenticated requests, the limit is
5,000 requests per hour. Make sure to handle the `RateLimitExceededException` in
your code to avoid issues when hitting the rate limit.

# Filtering Files by Extension

To filter files by specific extensions when using PyGithub, you can iterate
through the contents of the repository and check the file extension for each
file. Here's an example on how to do this:

```python
from github import Github


def get_filtered_files(repo, extensions):
    filtered_files = []
    contents = repo.get_contents("")

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file_extension = os.path.splitext(file_content.path)[1]
            if file_extension in extensions:
                filtered_files.append(file_content)

    return filtered_files


# Replace YOUR_ACCESS_TOKEN with the actual token you generated
access_token = "YOUR_ACCESS_TOKEN"
g = Github(access_token)

repo = g.get_repo("username/reponame")

# Specify the file extensions you want to filter
extensions = ['.md', '.py', '.yaml']

filtered_files = get_filtered_files(repo, extensions)

for file in filtered_files:
    print(file.path)
```

In this example, the function `get_filtered_files` takes a `repo` object and a
list of file extensions as input. It then iterates through the contents of the
repository, adding files with the specified extensions to the `filtered_files`
list.

Remember to replace `"YOUR_ACCESS_TOKEN"` with the actual token you generated
and `"username/reponame"` with the repository you want to access.

Note that you will need to import the `os` module for this code to work, so add
the following import statement at the beginning of your script:

```python
import os
```
# Decoding File Content

When you want to retrieve the contents of a file using
PyGithub, you can use the `get_contents()` method followed by
the `decoded_content` attribute and then decode the content using the "utf-8"
encoding.

Here's a complete example:

```python
from github import Github
import os

# Replace YOUR_ACCESS_TOKEN with the actual token you generated
access_token = "
User
why
do
I
need
utf - 8
ChatGPT
You
need
to
use
the
UTF - 8
encoding
to
properly
decode
the
content
of
the
file
because
the
`decoded_content`
attribute
returns
the
file
content as bytes.The
content
needs
to
be
decoded
into
a
string
to
be
human - readable and properly
processed in your
Python
script.

UTF - 8 is a
widely
used
encoding
for Unicode text that is compatible with ASCII.It is the default encoding for many programming languages, including Python, and is widely used on the web and in various data formats.By specifying the "utf-8" encoding, you ensure that the content is decoded correctly, preserving any special characters or non-ASCII symbols present in the file.

Here
's an example of how to decode file content using the "utf-8" encoding:

```python
file_content = repo.get_contents(file_path)
decoded_content = file_content.decoded_content.decode("utf-8")
print(decoded_content)
```

In this example, `repo` is the repository object, and `file_path` is the path of
the file you want to access. After decoding the content, you can print it or
manipulate it in your Python script as

Base64 is related to the process of decoding the content of a file in
PyGithub. When you call the `get_contents()` method, the content of the file is
returned as a base64 encoded string. Base64 is a binary-to-text encoding scheme
that is commonly used to represent binary data (like the content of a file) as
an ASCII string.

The `decoded_content` attribute in PyGithub is a helper attribute that
automatically decodes the base64 encoded content into bytes. After that, you
still need to decode the bytes into a human-readable string using an appropriate
character encoding like "utf-8".

Here's a summary of the steps:

1. `get_contents()` returns a base64 encoded string.
2. The `decoded_content` attribute decodes the base64 encoded string into bytes.
3. You use `decode("utf-8")` to convert the bytes into a human-readable string.

The following example shows the entire process:

```python
file_content = repo.get_contents(file_path)
decoded_content = file_content.decoded_content.decode("utf-8")
print(decoded_content)
```

In this example, `repo` is the repository object, and `file_path` is the path of
the file you want to access. After decoding the base64 encoded content and
converting the bytes into a UTF-8 encoded string, you can print it or manipulate
it in your Python script as needed.


Here's an example function that takes a `ContentFile` object as input and
returns the decoded content:

```python
from github import ContentFile


def get_decoded_content(content_file: ContentFile) -> str:
    if content_file.encoding == "base64":
        return content_file.decoded_content.decode("utf-8")
    elif content_file.encoding == "none":
        return content_file.content
    else:
        raise ValueError(f"Unsupported encoding: {content_file.encoding}")


# Usage example
content = get_decoded_content(self)
print(content)
```
