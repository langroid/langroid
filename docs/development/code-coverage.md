# Code coverage using coverage.py


You can use `coverage.py` locally to measure the code coverage
of your Python project.

To use `coverage.py` locally, follow these steps:

1. Install `coverage.py`: You can install `coverage.py` using pip:

```
pip install coverage
```

2. Run tests with `coverage.py`: To measure coverage while running your tests,
   use the `coverage run` command, followed by the command you normally use to
   run your tests. For example, if you use `pytest`, you would run:

```
coverage run -m pytest
```

Make sure to include the `--source` option to specify the package or folder
containing your source code. For example:

```
coverage run --source=my_package -m pytest
```

Replace `my_package` with the name of the package or folder containing your
source code.

3. Generate a coverage report: After running the tests with coverage, you can
   generate a report to see the coverage percentage. To do this, run:

```
coverage report
```

This will display a report in your terminal, including the coverage percentage
for each file in your project and an overall coverage percentage.

4. (Optional) Generate an HTML report: You can also generate an HTML report with
   more details, including highlighted source code showing which lines were
   covered by the tests. To generate an HTML report, run:

```
coverage html
```

This will create a folder named `htmlcov` in your project directory. Open
the `index.html` file in this folder using a web browser to view the detailed
HTML report.

Remember to run the tests with `coverage.py` and generate a report whenever you
make changes to your code to keep track of your code coverage.

# Coverage report from GitHub Actions

Yes, you can generate an HTML coverage report in a GitHub Actions workflow and
make it accessible from your README. One way to achieve this is by uploading the
generated report as a GitHub artifact. Here's how to do it:

1. Modify your GitHub Actions workflow file: Update your existing GitHub Actions
   workflow file (e.g., `.github/workflows/main.yml`) to generate an HTML
   coverage report and upload it as an artifact. Modify your workflow file as
   shown below:

```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ 3.7, 3.8, 3.9 ]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests with coverage
        run: |
          coverage run --source=my_package -m pytest
          coverage report
      - name: Generate HTML coverage report
        run: |
          coverage html
      - name: Upload coverage report as artifact
        uses: actions/upload-artifact@v2
        with:
          name: htmlcov
          path: htmlcov/
```

Replace `my_package` with the name of the package or folder containing your
source code.

This will generate an HTML coverage report and upload it as an artifact
named `htmlcov`.

2. Access the coverage report from the GitHub Actions page: After the workflow
   run completes, you can access the generated HTML coverage report from the
   GitHub Actions page. Go to the "Actions" tab in your repository, click on the
   latest workflow run, and find the "Artifacts" section at the bottom of the
   page. Download the `htmlcov` artifact and extract the contents to view the
   HTML coverage report locally.

3. Link the coverage report from your README: Unfortunately, you cannot directly
   link the generated HTML report from your README, as GitHub does not serve raw
   HTML files. However, you can provide instructions in your README on how to
   access the report from the GitHub Actions page, as explained in step 2.

Add the following text to your `README.md` file:

```
To access the latest HTML coverage report, go to the "Actions" tab in the 
repository, click on the latest workflow run, and download the `htmlcov` 
artifact from the "Artifacts" section at the bottom of the page. Extract the 
contents and open the `index.html` file to view the report.   
```

This will guide users on how to access the generated HTML coverage report for
your project.

# Create a private cogerage report page for your team

You're correct that GitHub Pages are public by default, which may not be
suitable for hosting sensitive information like code coverage reports for
private repositories.

As an alternative, you can use a separate private repository to store your
generated HTML coverage reports, and give access to your team members. Here's
how to do it:

1. Create a new private repository: Create a new private repository on GitHub,
   let's call it `your_coverage_reports`. Give access to your team members who
   should be able to view the coverage reports.

2. Set up a GitHub deploy key: In your main repository, go to the "Settings"
   tab, click on "Deploy keys" in the left sidebar, and click on "Add deploy
   key." Generate an SSH key pair using `ssh-keygen` and provide the public key
   as the deploy key. Make sure to check "Allow write access" before adding the
   key. Note the private key, as you will need it in the next step.

3. Modify your GitHub Actions workflow file: Update your existing GitHub Actions
   workflow file (e.g., `.github/workflows/main.yml`) to generate an HTML
   coverage report, commit it to the `your_coverage_reports` repository, and
   push the changes. Modify your workflow file as shown below:

```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ 3.7, 3.8, 3.9 ]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests with coverage
        run: |
          coverage run --source=my_package -m pytest
          coverage report
      - name: Generate HTML coverage report
        run: |
          coverage html
      - name: Commit and push coverage report to your_coverage_reports repo
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
        run: |
          git config --global user.name "GitHub Actions"
          git config --global user.email "actions@github.com"
          echo "$DEPLOY_KEY" > deploy_key.pem
          chmod 600 deploy_key.pem
          git clone git@github.com:your_username/your_coverage_reports.git --branch main coverage_reports
          cp -r htmlcov/* coverage_reports/
          cd coverage_reports
          git add .
          git commit -m "Update coverage report"
          git push origin main
```

Replace `my_package` with the name of the package or folder containing your
source code, and `your_username` with your GitHub username.

4. Add the deploy key to GitHub Secrets: In your main repository, go to the "
   Settings" tab, click on "Secrets," and click on "New repository secret." Name
   the secret `DEPLOY_KEY` and paste the private key from the SSH key pair you
   generated in step 2 as the value.

5. Link the coverage report from your README: In your `README.md` file, add the
   following text:

```
To access the latest HTML coverage report, go to the [your_coverage_reports](https://github.com/your_username/your_coverage_reports) repository.
```

Replace `your_username` with your GitHub username.

Now, when your GitHub Actions workflow runs, it will generate an HTML coverage
report, commit it to the `your_coverage_reports`

# Coverage badge using Codecov

Here are the detailed steps to set up `coverage.py` in your Python repository
using GitHub Actions and to display a code coverage badge in your README file.

1. Install `coverage.py`: Add `coverage` to your project's requirements. You can
   add it to your `requirements.txt` or `Pipfile` (if you're using Pipenv) or to
   your `pyproject.toml` file (if you're using Poetry). For example,
   in `requirements.txt`, add:

```
coverage
```

2. Modify your GitHub Actions workflow file: You need to update your existing
   GitHub Actions workflow file (e.g., `.github/workflows/main.yml`) to run
   tests using `coverage.py` and generate a coverage report. Modify your
   workflow file as shown below:

```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ 3.7, 3.8, 3.9 ]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests with coverage
        run: |
          coverage run --source=my_package -m pytest
          coverage report
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v2
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

Replace `my_package` with the name of the package or folder containing your
source code.

3. Sign up for Codecov: Go to https://codecov.io/ and sign up using your GitHub
   account. Follow the instructions to add your repository.

4. Add Codecov token to GitHub Secrets: After adding your repository, Codecov
   will provide you with an API token. Add this token to your GitHub
   repository's secrets. Go to your repository's Settings tab, click on Secrets,
   and then click on "New repository secret." Name the secret `CODECOV_TOKEN`
   and paste the token as the value.

5. Add the coverage badge to your README: In your `README.md` file, add the
   following line at the top:

```
[![codecov](https://codecov.io/gh/your_username/your_repository/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/your_username/your_repository)
```

Replace `your_username` with your GitHub username, `your_repository` with your
repository's name, and `YOUR_CODECOV_TOKEN` with the token provided by Codecov.

Now, whenever you push changes to your repository, GitHub Actions will run the
tests using `coverage.py`, generate a coverage report, and upload it to Codecov.
The coverage badge in your README will display the percentage of code covered by
tests.

# Coverage badge for private GitHub repositories

If you want to display a code coverage
badge for a private GitHub repository without using a paid service, you can use
the `shields.io` service to create a custom badge. It won't update automatically
with every commit, but you can manually update the badge whenever you update
your code coverage.

Here's how to create a custom badge for your code coverage:

1. Generate a code coverage report using `coverage.py`: After running your tests
   using `coverage.py`, you'll have a coverage report. You can generate the
   report as a percentage by running:

```
coverage report -m
```

You will see a line similar to the following at the end of the output:

```
TOTAL                      95     5    95%
```

In this example, the code coverage is 95%.

2. Create a custom badge with `shields.io`: Visit the following URL and
   replace `95` with your code coverage percentage:

```
https://img.shields.io/badge/coverage-95%25-success
```

This will generate a custom badge with the specified code coverage. The last
part of the URL (`success`) determines the color of the badge. You can
use `success`, `important`, `critical`, `informational`, or `inactive` depending
on the coverage percentage or your preference.

3. Add the custom badge to your README: Add the following line to
   your `README.md` file, replacing the URL with the one you generated in step
   2:

```
![Coverage](https://img.shields.io/badge/coverage-95%25-success)
```

This will display the custom code coverage badge in your README.

Please note that this method requires manual updates to the badge whenever your
code coverage changes. It doesn't automatically update the badge with every
commit like other services do.