"""
The LanceRAGTaskCreator.new() method creates a 3-Agent system that uses this agent.
It takes a LanceDocChatAgent instance as argument, and adds two more agents:
- LanceQueryPlanAgent, which is given the LanceDB schema in LanceDocChatAgent,
and based on this schema, for a given user query, creates a Query Plan
using the QueryPlanTool, which contains a filter, a rephrased query,
and a dataframe_calc.
- QueryPlanCritic, which is given the LanceDB schema in LanceDocChatAgent,
 and gives feedback on the Query Plan and Result using the QueryPlanFeedbackTool.

The LanceRAGTaskCreator.new() method sets up the given LanceDocChatAgent and
QueryPlanCritic as sub-tasks of the LanceQueryPlanAgent's task.

Langroid's built-in task orchestration ensures that:
- the LanceQueryPlanAgent reformulates the plan based
    on the QueryPlanCritics's feedback,
- LLM deviations are corrected via tools and overrides of ChatAgent methods.
"""

import logging

from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.agent.special.lance_rag.critic_agent import (
    QueryPlanCritic,
    QueryPlanCriticConfig,
)
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgent,
    LanceQueryPlanAgentConfig,
)
from langroid.agent.task import Task
from langroid.mytypes import Entity

logger = logging.getLogger(__name__)


class LanceRAGTaskCreator:
    @staticmethod
    def new(
        agent: LanceDocChatAgent,
        interactive: bool = True,
    ) -> Task:
        """
        Add a LanceFilterAgent to the LanceDocChatAgent,
        set up the corresponding Tasks, connect them,
        and return the top-level query_plan_task.
        """
        doc_agent_name = "LanceRAG"
        critic_name = "QueryPlanCritic"
        query_plan_agent_config = LanceQueryPlanAgentConfig(
            critic_name=critic_name,
            doc_agent_name=doc_agent_name,
            doc_schema=agent._get_clean_vecdb_schema(),
            llm=agent.config.llm,
        )
        query_plan_agent_config.set_system_message()

        critic_config = QueryPlanCriticConfig(
            doc_schema=agent._get_clean_vecdb_schema(),
            llm=agent.config.llm,
        )
        critic_config.set_system_message()

        query_planner = LanceQueryPlanAgent(query_plan_agent_config)
        query_plan_task = Task(
            query_planner,
            interactive=interactive,
        )
        critic_agent = QueryPlanCritic(critic_config)
        critic_task = Task(
            critic_agent,
            interactive=False,
        )
        rag_task = Task(
            agent,
            name="LanceRAG",
            interactive=False,
            done_if_response=[Entity.LLM],  # done when non-null response from LLM
            done_if_no_response=[Entity.LLM],  # done when null response from LLM
        )
        query_plan_task.add_sub_task([critic_task, rag_task])
        return query_plan_task
