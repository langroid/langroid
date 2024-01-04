import logging

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.lance_rag.lance_tools import (
    QueryPlanFeedbackTool,
    QueryPlanTool,
)
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgentConfig,
)
from langroid.mytypes import Entity
from langroid.utils.constants import DONE, PASS

logger = logging.getLogger(__name__)


class QueryPlanCriticConfig(LanceQueryPlanAgentConfig):
    name = "QueryPlanCritic"
    system_message = """
    You are an expert at carefully planning a query that needs to be answered
    based on a large collection of documents. These docs have a special `content` field
    and additional FILTERABLE fields in the SCHEMA below:
    
    {doc_schema}
    
    You will receive a QUERY PLAN consisting of a 
    ORIGINAL QUERY, SQL-Like FILTER, REPHRASED QUERY, 
    a DATAFRAME CALCULATION, and a "result" field which is the 
    answer received from an assistant that used this QUERY PLAN.
    
    Your job is to act as a CRITIC and provide feedback, 
    ONLY using the `query_plan_feedback` tool, and DO NOT SAY ANYTHING ELSE.
    You must take `result` field into account
    and judge whether it is a reasonable answer, and accordingly give your feedback.
    
    VERY IMPORTANT: IF THE RESULT (ANSWER) seems reasonable, then you should consider
    the query plan to be fine, and only ask for a revision if you notice 
    something obviously wrong with the query plan.
    Typically, when there is a non-empty answer, 
    you SHOULD NOT ASK FOR A REVISION (i.e. set `feedback` field as empty string),
    unless there is some glaring problem.
    
    OTHERWISE:
        
    When giving feedback, SUGGEST CHANGES TO:
    - the FILTER, if it appears too restrictive, e.g. prefer "LIKE" over "=", 
        e.g. "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'" 
    - the QUERY (i.e. possibly REPHRASED QUERY), and/or 
    - the DATAFRAME CALCULATION, if any
     
    Keep these in mind:
    * The FILTER must only use fields in the SCHEMA above, EXCEPT `content`
    * The FILTER can be improved by RELAXING it, e.g. using "LIKE" instead of "=",
        e.g. "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'" 
    * The DATAFRAME CALCULATION must only use fields in the SCHEMA above. 
    * The REPHRASED QUERY should NOT refer to any FILTER fields, and should
        make sense with respect to the intended purpose, i.e. to be used to 
        MATCH the CONTENT of the docs.
    * The ASSISTANT does NOT know anything about the FILTER fields
    * The DATAFRAME CALCULATION, if any, should be suitable to answer 
       the user's ORIGINAL QUERY.
    
    ALWAYS use `query_plan_feedback` tool/fn to present your feedback!
    and DO NOT SAY ANYTHING ELSE OUTSIDE THE TOOL/FN.
        
    """


def plain_text_query_plan(msg: QueryPlanTool) -> str:
    plan = f"""
    OriginalQuery: {msg.original_query}
    Filter: {msg.filter}
    Query: {msg.query}
    DataframeCalc: {msg.dataframe_calc}
    """
    if msg.result is not None and msg.result != "":
        plan += f"\nResult: {msg.result}"
    return plan


class QueryPlanCritic(ChatAgent):
    """
    Critic for LanceFilterAgent.
    """

    def __init__(self, cfg: LanceQueryPlanAgentConfig):
        super().__init__(cfg)
        self.config = cfg
        self.curr_query_plan: QueryPlanTool | None = None
        self.enable_message(QueryPlanTool, use=False, handle=True)
        self.enable_message(QueryPlanFeedbackTool, use=True, handle=True)

    def query_plan(self, msg: QueryPlanTool) -> str:
        """Present query plan in plain text (not JSON) so LLM can give feedback"""
        self.curr_query_plan = msg
        return plain_text_query_plan(msg)

    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> str:
        """Format Valid so return to Query Planner"""
        return DONE + " " + PASS  # return to Query Planner

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Create QueryPlanFeedbackTool since LLM forgot"""
        if isinstance(msg, ChatDocument):
            if msg.metadata.sender == Entity.LLM:
                # our LLM forgot to use the QueryPlanFeedbackTool
                feedback = QueryPlanFeedbackTool(feedback=msg.content)
                msg.tool_messages = [feedback]
                msg.content = DONE
                return msg
            elif msg.metadata.sender == Entity.USER:
                # parent task forgot to use the QueryPlanTool
                # so make a special QueryPlanFeedbackTool
                feedback = QueryPlanFeedbackTool(
                    feedback="""
                    You forgot to send a query plan using the `query_plan` tool.
                    """
                )
                msg.tool_messages = [feedback]
                msg.content = DONE
                return msg
        return None
