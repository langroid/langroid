"""
Data Analysis Workflow Demo
===========================

This demo showcases a multi-agent data analysis workflow using Langroid that automates
the complete data science pipeline from data cleaning to visualization and regression analysis.

## What this demo does:

The workflow consists of three specialized AI agents that work together:

1. **DataCleaner**: Cleans and preprocesses raw tabular data
   - Handles missing values, data type conversions, outlier detection
   - Uses pandas operations to prepare data for analysis

2. **DataVisualizer**: Creates data visualizations using Matplotlib
   - Generates plots, charts, and graphs to explore data patterns
   - Can visualize both raw data and model results

3. **DataRegressor**: Performs machine learning analysis
   - Supports both regression and classification tasks
   - Uses scikit-learn models like LinearRegression, LogisticRegression, DecisionTree

## How to run
python .\examples\data-qa\data-analysis-demo\data_analysis_workflow.py

# Command Line Options:
- `--debug`: Enable debug mode for detailed logging
- `--nostream`: Disable streaming output
- `--nocache`: Disable caching of results
- `--model`: Specify the language model to use (default is GPT-4o)



### Interactive workflow:
1. When prompted, enter a dataset URL or local file path (or press Enter for default)
2. The MainAgent will coordinate the workflow and ask you what analysis you want
3. Interact with the agents by describing your data analysis goals
4. The agents will automatically clean data, create visualizations, and run ML models

### Example datasets to try:
- Default: Airline safety data (built-in)
- Any CSV file accessible via URL or local path
- Try: "https://raw.githubusercontent.com/datasets/csv-examples/master/data/simple.csv"

The demo uses OpenAI's GPT models by default. Make sure to set your OPENAI_API_KEY 
environment variable before running.
"""

import langroid as lr
import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from data_visualization_agent import (
    DataVisualizationAgent,
    DataVisualizationConfig,
    DataVisualizationTool,
    ResultsHelperTool,
)
from data_processing_agent import (
    DataProcessingAgent,
    DataProcessingConfig,
    PandasProcessTool,
)
from data_regression_agent import (
    DataRegressionAgent,
    DataRegressionTool,
    DataRegressionConfig,
)
from langroid.utils.configuration import set_global, Settings
from utils import SharedData
from langroid.utils.constants import DONE

import typer
from rich.prompt import Prompt
from messages import (
    DEFAULT_DATA_PREPROCESSING_SYSTEM_MESSAGE,
    DEFAULT_DATA_VISUALIZATION_SYSTEM_MESSAGE,
    DEFAULT_DATA_REGRESSION_SYSTEM_MESSAGE,
)

DEFAULT_DATASET_PATH = "https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv"

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode", is_flag=True
    ),
    no_stream: bool = typer.Option(
        False, "--nostream", "-ns", help="Disable streaming", is_flag=True
    ),
    nocache: bool = typer.Option(
        False, "--nocache", "-nc", help="Disable cache", is_flag=True
    ),
    model: str = typer.Option("", "--model", "-m", help="Model name"),
) -> None:
    """
    Main entry point for the data analysis demo.
    Sets up agents for data cleaning, visualization, and regression.
    """
    # Set global configuration
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )

    # Prompt user for dataset path or URL
    data_input: str = Prompt.ask(
        "[blue]Enter a local path or URL to a tabular dataset (hit enter to use default)\n",
        default=DEFAULT_DATASET_PATH,
    )
    shared_data: SharedData = SharedData(data_input)

    # Configure the language model
    llm_cfg = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=16_000,
        timeout=45,
        temperature=0.2,
    )

    # Main agent manages the workflow
    main_agent_cfg = ChatAgentConfig(
        name="MainAgent",
        llm=llm_cfg,
        system_message=f"""
        You will manage a data analysis workflow with the following components:
        1. Data Cleanup: Clean and preprocess the data.
        2. Data Visualization: Visualize the data using Matplotlib.
        3. Data Regression: Perform regression analysis on the data.
        You will tell the DataCleaner to clean the data.
        You will tell the DataVisualizer to visualize the data.
        You will tell the DataRegressor to perform regression analysis.

        STATE THE NAME OF THE AGENT you want to utilize and the context that might be necessary for the task, if it is available, potentially returned by another agent.
        Don't ask the user for context, use the information agents have provided you, and use it to inform your decisions.
        Context can include information about the data, such as column names, types, correlations, or any other relevant details that will help the agent perform its task effectively.

        In the form:

        <AgentName>: <context>

        You can use human input or your best judgement to determine the next steps in the workflow, and answer any questions the user may have.
        It is your job to CALL THE APPROPRIATE AGENT for each task.

        CALL ONE AGENT AT A TIME, and wait for its response before proceeding to the next step.

        Do NOT ask the user for data input. 
        Once the workflow is complete, say {DONE} to indicate completion.
        """,
    )
    main_agent = ChatAgent(main_agent_cfg)
    main_task = lr.Task(
        main_agent,
        name="MainTask",
        llm_delegate=True,
        single_round=False,
        interactive=True,
    )

    # Data cleaning agent
    data_cleanup_agent_cfg = DataProcessingConfig(
        name="DataCleaner",
        data=shared_data,
        llm=llm_cfg,
        system_message=DEFAULT_DATA_PREPROCESSING_SYSTEM_MESSAGE,
        full_eval=True,
    )
    data_cleanup_agent = DataProcessingAgent(data_cleanup_agent_cfg)
    data_cleanup_task = lr.Task(
        data_cleanup_agent,
        name="DataCleanupTask",
        llm_delegate=True,
        single_round=False,
        interactive=True,
    )

    # Data visualization agent
    data_visualization_agent_cfg = DataVisualizationConfig(
        name="DataVisualizer",
        data=shared_data,
        llm=llm_cfg,
        system_message=DEFAULT_DATA_VISUALIZATION_SYSTEM_MESSAGE,
        full_eval=False,
    )
    data_visualization_agent = DataVisualizationAgent(data_visualization_agent_cfg)
    data_visualization_task = lr.Task(
        data_visualization_agent,
        name="DataVisualizationTask",
        llm_delegate=True,
        single_round=False,
        interactive=True,
    )

    # Data regression agent
    data_regression_agent_cfg = DataRegressionConfig(
        data=shared_data,
        llm=llm_cfg,
        system_message=DEFAULT_DATA_REGRESSION_SYSTEM_MESSAGE,
        full_eval=False,
    )
    data_regression_agent = DataRegressionAgent(data_regression_agent_cfg)
    data_regression_task = lr.Task(
        data_regression_agent,
        name="DataRegressionTask",
        llm_delegate=True,
        single_round=False,
        interactive=True,
    )

    # Add sub-tasks to the main task
    main_task.add_sub_task(
        [data_cleanup_task, data_visualization_task, data_regression_task]
    )

    # Enable tool messages for agents
    data_cleanup_agent.enable_message(PandasProcessTool)
    data_visualization_agent.enable_message(DataVisualizationTool)
    data_visualization_agent.enable_message(ResultsHelperTool)
    data_regression_agent.enable_message(DataRegressionTool)

    # Start the main task (interactive loop)
    main_task.run()


if __name__ == "__main__":
    app()
