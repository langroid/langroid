# Running Code in a Container

Suppose you want to write a Python function that takes a string containing
Python code, and runs that code in a container.

Creating a method that runs arbitrary Python code, especially if that code can
access objects and data from the class, is a complex and potentially dangerous
task, depending on the source and the nature of the code being run. However, for
this task, we will use the Docker API for Python to create a Docker container
that can run Python code.

Please note that this approach still has a lot of potential security issues,
especially when the Python code comes from untrusted sources, so you should take
additional precautions depending on your specific application.

This solution also assumes that you have Docker installed and running on your
machine.

Here is a sample of how this might look like:

```python
import docker
from docker.types import Mount


class MessageHandler:

    def __init__(self):

    self.data_file = '/path/to/data_file.csv'


# add more class attributes as needed

def run_python(self, code: str):
    client = docker.from_env()
    # specify the directory where your data files are as the source.
    mount = Mount("/data", "/host/data/path", type="bind")
    try:
        # This assumes you have a Python Docker image called "python" locally.
        # If not, Docker will attempt to pull it from Docker Hub.
        # Make sure to adjust the Python version based on your requirements.
        container = client.containers.run(
            "python:3.8",
            command=["python", "-c", code],
            mounts=[mount],
            working_dir="/data",
            remove=True,
            detach=True
        )
        # Get the output from the container
        output = container.logs()
        return output.decode('utf-8')
    except docker.errors.ContainerError as e:
        print(f'Error while running container: {e}')


# usage
handler = MessageHandler()
code = """
with open('data_file.csv', 'r') as file:
print(file.read())
"""
print(handler.run_python(code))
```

This code works by creating a Docker container that runs the Python interpreter
on the passed code string. We bind-mount the directory containing the data files
to `/data` in the Docker container, so the Python code can read/write files
from/to that directory.

Please take note that the Docker container does not have access to the Python
objects in your running Python program. If you need to pass data to the Python
code running in the container, you need to serialize it to a file, pass the file
to the container, and then deserialize it in the container.

Again, caution should be used with this approach. Always validate and sanitize
any input that will be run as code to prevent any potential security issues.

# Using a base image

What about ensuring required python pkgs are available in the container?

A common practice is to use a base Docker image, and then add the Python
packages your program requires. You can do this by creating your own Docker
image that extends the base Python image and installs the necessary packages.
Here's a simple Dockerfile that does this:

```dockerfile
# Start with the Python 3.8 image
FROM python:3.8

# Add metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Python with some common packages"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install any required packages
RUN pip install pandas numpy scipy sklearn

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
```

You can build this Docker image with a command
like `docker build -t my-python:1.0 .`.

Then in your `MessageHandler` class, you can use "my-python:1.0" as the image to
run the containers:

```python
container = client.containers.run(
    "my-python:1.0",
    command=["python", "-c", code],
    mounts=[mount],
    working_dir="/data",
    remove=True,
    detach=True
)
```

This way, every time you run Python code in a container, it will have access to
the packages pandas, numpy, scipy, and sklearn.

Please note, creating Docker images for each Python environment you need to
support can be a time-consuming task. However, it's a task that you typically
need to do only once for each environment, and it's a reliable way to ensure
that your Python code always has the packages it needs, regardless of what
computer it's run on.

# Returning results from code running in a container

There are multiple ways to return results from a container depending on the type
of result you expect:

1. **Primitive Data Types (Strings, Numbers, etc.)**: If your code produces
   strings or numbers, you can simply print these values, and then capture the
   stdout logs from the container, as shown in the previous examples. The
   printed values can be returned from the `run_python` function.

2. **Objects, Complex Data Types**: If your code produces Python objects or
   complex data types, you can serialize these objects to a file using something
   like JSON or `pickle`, and then read this file back in your host application.

3. **Files**: If your code generates files, you can write these files to the
   mounted directory (`/data` in the previous examples). After the code has been
   executed, these files will remain in the host directory, and their path can
   be returned from the `run_python` function.

Here's an example where `run_python` returns a path to a file that is generated
by the executed code:

```python
class MessageHandler:
    def __init__(self):
        self.data_file = '/path/to/data_file.csv'
        # add more class attributes as needed

    def run_python(self, code: str, packages: list):
        client = docker.from_env()
        mount = Mount("/data", "/host/data/path", type="bind")
        install_packages_code = f'import sys, subprocess; [subprocess.check_call([sys.executable, "-m", "pip", "install", pkg]) for pkg in {packages}]'
        try:
            container = client.containers.run(
                "python:3.8",
                command=["python", "-c", f"{install_packages_code}; {code}"],
                mounts=[mount],
                working_dir="/data",
                remove=True,
                detach=True
            )
            # For this example, we'll assume that the code writes its output to a file called 'output.txt'.
            return '/host/data/path/output.txt'
        except docker.errors.ContainerError as e:
            print(f'Error while running container: {e}')


# usage
handler = MessageHandler()
code = """
with open('output.txt', 'w') as f:
    f.write('Hello, World!')
"""
packages = []
output_file = handler.run_python(code, packages)
print(f'Results are stored in: {output_file}')
```

In this example, the Python code writes its result to a file `output.txt`, and
the `run_python` function returns the path to this file on the host machine.

# Using JSON to serilize/deserialize Python objects to return from the container

Yes, you can use JSON to dump results to the log, and then parse the output from
the container log. This would work well for primitive data types and simple data
structures like lists and dictionaries. Here's an example:

```python
import json


class MessageHandler:
    def __init__(self):
        self.data_file = '/path/to/data_file.csv'
        # add more class attributes as needed

    def run_python(self, code: str, packages: list):
        client = docker.from_env()
        mount = Mount("/data", "/host/data/path", type="bind")
        install_packages_code = f'import sys, subprocess; [subprocess.check_call([sys.executable, "-m", "pip", "install", pkg]) for pkg in {packages}]'
        try:
            container = client.containers.run(
                "python:3.8",
                command=["python", "-c", f"{install_packages_code}; {code}"],
                mounts=[mount],
                working_dir="/data",
                remove=True,
                detach=True
            )
            output = container.logs().decode('utf-8')
            # Parse the output as JSON
            result = json.loads(output)
            return result
        except docker.errors.ContainerError as e:
            print(f'Error while running container: {e}')


# usage
handler = MessageHandler()
code = """
import json
data = {'foo': 42, 'bar': [1, 2, 3]}
print(json.dumps(data))
"""
packages = []
result = handler.run_python(code, packages)
print(f'Result: {result}')  # prints: Result: {'foo': 42, 'bar': [1, 2, 3]}
```

In this example, the Python code creates a dictionary, converts it to a JSON
string using `json.dumps`, and prints it. The `run_python` function then
captures the container log, which is the printed JSON string, and parses it back
into a Python object using `json.loads`.

Please note that this approach can only be used for serializable objects that
can be converted to a JSON string. If your code produces non-serializable
objects or objects that cannot be represented in JSON (like custom classes or
complex data structures), you'll need to use a different approach (like
serializing to a file using `pickle`, as discussed in a previous comment).