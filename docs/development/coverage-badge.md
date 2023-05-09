
GitHub Pages are public by default. If you want to keep your
private repo's information private, you can use create a coverage badge
by storing the coverage information as a GitHub Artifact. Here's how you can 
do it: 

1. **Create a JSON file for coverage data**: You already have this step covered
   since you are generating the JSON coverage report with `--cov-report json`.

2. **Upload coverage data as a GitHub Artifact**: Modify your GitHub workflow to
   upload the generated JSON coverage report as a GitHub Artifact. Add the
   following steps to your workflow:

   ```yaml
   - name: Archive coverage report
     uses: actions/upload-artifact@v2
     with:
       name: coverage-report
       path: coverage.json
   ```

3. **Create a GitHub Action to update the badge**: Create a new GitHub Action
   that will read the coverage percentage from the JSON report in the Artifact,
   and then update the badge accordingly.

   First, you'll need to create a separate workflow file (
   e.g., `update_badge.yml`) in your `.github/workflows` folder with the
   following content:

   ```yaml
   name: Update Coverage Badge
   on:
     workflow_run:
       workflows:
         - <your_main_workflow_name>
       types:
         - completed

   jobs:
     update_badge:
       runs-on: ubuntu-latest
       steps:
       - name: Download coverage report
         uses: actions/github-script@v5
         with:
           script: |
             const fs = require('fs');
             const artifact = await github.rest.actions.listWorkflowRunArtifacts({
                owner: context.repo.owner,
                repo: context.repo.repo,
                run_id: context.payload.workflow_run.id,
             });
             const coverageArtifact = artifact.data.artifacts.find(a => a.name === 'coverage-report');
             const coverageDownload = await github.rest.actions.downloadArtifact({
                owner: context.repo.owner,
                repo: context.repo.repo,
                artifact_id: coverageArtifact.id,
                archive_format: 'zip',
             });
             fs.writeFileSync('coverage-report.zip', Buffer.from(coverageDownload.data));
       - name: Extract coverage report
         run: |
           unzip coverage-report.zip
   ```

   Replace `<your_main_workflow_name>` with the name of the workflow that runs
   your tests with coverage.

   Next, you'll need to extract the coverage percentage from the JSON report and
   update the badge using the `shields.io` API. Add these steps to
   the `update_badge.yml` file:

   ```yaml
       - name: Install Node.js
         uses: actions/setup-node@v2
         with:
           node-version: 14

       - name: Install dependencies
         run: |
           npm install

       - name: Update coverage badge
         run: |
           node updateBadge.js
   ```

4. **Create a script to update the badge**: Create a new script
   called `updateBadge.js` in your repository with the following content:

   ```javascript
   const fs = require('fs');
   const axios = require('axios');

   const coverageReport = JSON.parse(fs.readFileSync('coverage.json', 'utf-8'));
   const coveragePercentage = coverageReport.total.lines.percent.toFixed(2);

   const badgeUrl = `https://img.shields.io/badge/coverage-${coveragePercentage}%25-green?style=flat-square&logo=python`;

   axios.get(badgeUrl, { responseType: 'arraybuffer' })
     .then((response) => {
       fs.writeFileSync('coverage_badge.svg', Buffer.from(response.data));
     })
     .catch((error) => {
       console.error('Failed to update coverage badge:', error.message);
       process.exit(1);
     });
   ``