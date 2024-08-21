"""
LanceQueryPlanAgent is a ChatAgent created with a specific document schema.
Given a QUERY, the LLM constructs a Query Plan consisting of:
- filter condition if needed (or empty string if no filter is needed)
- query - a possibly rephrased query that can be used to match the `content` field
- dataframe_calc - a Pandas-dataframe calculation/aggregation string, possibly empty
- original_query - the original query for reference

This agent has access to two tools:
- QueryPlanTool, which is used to generate the Query Plan, and the handler of
    this tool simply passes it on to the RAG agent named in config.doc_agent_name.
- QueryPlanFeedbackTool, which is used to handle feedback on the Query Plan and
  Result from the RAG agent. The QueryPlanFeedbackTool is used by
  the QueryPlanCritic, who inserts feedback into the `feedback` field
"""

import logging
from typing import Optional

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.lance_tools import (
    AnswerTool,
    QueryPlan,
    QueryPlanAnswerTool,
    QueryPlanFeedbackTool,
    QueryPlanTool,
)
from langroid.agent.tools.orchestration import AgentDoneTool, ForwardTool
from langroid.utils.constants import NO_ANSWER

logger = logging.getLogger(__name__)


class LanceQueryPlanAgentConfig(ChatAgentConfig):
    name: str = "LancePlanner"
    critic_name: str = "QueryPlanCritic"
    doc_agent_name: str = "LanceRAG"
    doc_schema: str = ""
    use_tools = False
    max_retries: int = 5  # max number of retries for query plan
    use_functions_api = True

    system_message = """
    You will receive a QUERY, to be answered based on an EXTREMELY LARGE collection
    of documents you DO NOT have access to, but your ASSISTANT does.
    You only know that these documents have a special `content` field
    and additional FILTERABLE fields in the SCHEMA below, along with the 
    SAMPLE VALUES for each field, and the DTYPE in PANDAS TERMINOLOGY.
    
    {doc_schema}
    
    Based on the QUERY and the above SCHEMA, your task is to determine a QUERY PLAN,
    consisting of:
    -  a PANDAS-TYPE FILTER (can be empty string) that would help the ASSISTANT to 
        answer the query.
        Remember the FILTER can refer to ANY fields in the above SCHEMA
        EXCEPT the `content` field of the documents. 
        ONLY USE A FILTER IF EXPLICITLY MENTIONED IN THE QUERY.
        TO get good results, for STRING MATCHES, consider using LIKE instead of =, e.g.
        "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'"
        YOUR FILTER MUST BE A PANDAS-TYPE FILTER, respecting the shown DTYPES.        
    - a possibly REPHRASED QUERY (CANNOT BE EMPTY) to be answerable given the FILTER.
        Keep in mind that the ASSISTANT does NOT know anything about the FILTER fields,
        so the REPHRASED QUERY should NOT mention ANY FILTER fields.
        The assistant will answer based on documents whose CONTENTS match the QUERY, 
        possibly REPHRASED. 
        !!!!****THE REPHRASED QUERY SHOULD NEVER BE EMPTY****!!!
    - an OPTIONAL SINGLE-LINE Pandas-dataframe calculation/aggregation string 
        that can be used to calculate the answer to the original query, 
        e.g. "df["rating"].mean()",
        or "df.groupby("director").mean()["rating"]", 
        or EMPTY string if no calc is needed. 
        The dataframe calc CAN refer to the `content` field.
        If a DataFrame calculation is NOT needed, leave this field EMPTY.
        
        IMPORTANT: The DataFrame `df` in this calculation is the result of 
        applying the FILTER AND REPHRASED QUERY to the documents.
        
        WATCH OUT!! When deciding the dataframe calc, if any, CAREFULLY
        note what the query is asking, and ensure that the result of your
        dataframe calc expression would answer the query.                
    
    
    EXAMPLE:
    ------- 
    Suppose there is a document-set about crime reports, where:
     CONTENT = crime report,
     Filterable SCHEMA consists of City, Year, num_deaths.
    
    Then given this ORIGINAL QUERY: 
    
        Total deaths in shoplifting crimes in Los Angeles in 2023?
    
    A POSSIBLE QUERY PLAN could be:
    
    FILTER: "City LIKE '%Los Angeles%' AND Year = 2023"
    REPHRASED QUERY: "shoplifting crime" --> this will be used to MATCH content of docs
         [NOTE: we dropped the FILTER fields City and Year since the 
         ASSISTANT does not know about them and only uses the query to 
         match the CONTENT of the docs.]
    DATAFRAME CALCULATION: "df["num_deaths"].sum()"
        NOTE!!! The DataFrame `df` in this calculation is the result of
        applying the FILTER AND REPHRASED QUERY to the documents, 
        hence this computation will give the total deaths in shoplifting crimes.
    ------------- END OF EXAMPLE ----------------
    
    The FILTER must be a PANDAS-like condition, e.g. 
    "year > 2000 AND genre = 'ScienceFiction'".
    To ensure you get useful results, you should make your FILTER 
    NOT TOO STRICT, e.g. look for approximate match using LIKE, etc.
    E.g. "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'"
    Use DOT NOTATION to refer to nested fields, e.g. `metadata.year`, etc. 
        
    You must FIRST present the QUERY PLAN using the `query_plan` tool/function.
    This will be handled by your document assistant, who will produce an ANSWER.
            
    You may receive FEEDBACK on your QUERY PLAN and received ANSWER,
    from the 'QueryPlanCritic' who may offer suggestions for
    a better FILTER, REPHRASED QUERY, or DATAFRAME CALCULATION.
                  
    At the BEGINNING if there is no query, ASK the user what they want to know.
    """

    def set_system_message(self) -> None:
        self.system_message = self.system_message.format(
            doc_schema=self.doc_schema,
        )


class LanceQueryPlanAgent(ChatAgent):
    def __init__(self, config: LanceQueryPlanAgentConfig):
        super().__init__(config)
        self.config: LanceQueryPlanAgentConfig = config
        # This agent should generate the QueryPlanTool
        # as well as handle it for validation
        self.enable_message(QueryPlanTool, use=True, handle=True)
        self.enable_message(QueryPlanFeedbackTool, use=False, handle=True)
        self.enable_message(AnswerTool, use=False, handle=True)
        # neither use nor handle! Added to "known" tools so that the Planner agent
        # can avoid processing it
        self.enable_message(QueryPlanAnswerTool, use=False, handle=False)
        # LLM will not use this, so set use=False (Agent generates it)
        self.enable_message(AgentDoneTool, use=False, handle=True)

    def init_state(self) -> None:
        super().init_state()
        self.curr_query_plan: QueryPlan | None = None
        self.expecting_query_plan: bool = False
        # how many times re-trying query plan in response to feedback:
        self.n_retries: int = 0
        self.n_query_plan_reminders: int = 0
        self.result: str = ""  # answer received from LanceRAG

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        self.expecting_query_plan = True
        return super().llm_response(message)

    def query_plan(self, msg: QueryPlanTool) -> ForwardTool | str:
        """Valid, tool msg, forward chat_doc to RAG Agent.
        Note this chat_doc will already have the
        QueryPlanTool in its tool_messages list.
        We just update the recipient to the doc_agent_name.
        """
        # save, to be used to assemble QueryPlanResultTool
        if len(msg.plan.dataframe_calc.split("\n")) > 1:
            return "DATAFRAME CALCULATION must be a SINGLE LINE; Retry the `query_plan`"
        self.curr_query_plan = msg.plan
        self.expecting_query_plan = False

        # To forward the QueryPlanTool to doc_agent, we could either:

        # (a) insert `recipient` in the QueryPlanTool:
        # QPWithRecipient = QueryPlanTool.require_recipient()
        # qp = QPWithRecipient(**msg.dict(), recipient=self.config.doc_agent_name)
        # return qp
        #
        # OR
        #
        # (b) create an agent response with recipient and tool_messages.
        # response = self.create_agent_response(
        #     recipient=self.config.doc_agent_name, tool_messages=[msg]
        # )
        # return response

        # OR
        # (c) use the ForwardTool:
        return ForwardTool(agent=self.config.doc_agent_name)

    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> str | AgentDoneTool:
        """Process Critic feedback on QueryPlan + Answer from RAG Agent"""
        # We should have saved answer in self.result by this time,
        # since this Agent seeks feedback only after receiving RAG answer.
        if (
            msg.suggested_fix == ""
            and NO_ANSWER not in self.result
            and self.result != ""
        ):
            # This means the result is good AND Query Plan is fine,
            # as judged by Critic
            # (Note sometimes critic may have empty suggested_fix even when
            # the result is NO_ANSWER)
            self.n_retries = 0  # good answer, so reset this
            return AgentDoneTool(content=self.result)
        self.n_retries += 1
        if self.n_retries >= self.config.max_retries:
            # bail out to avoid infinite loop
            self.n_retries = 0
            return AgentDoneTool(content=NO_ANSWER)

        # there is a suggested_fix, OR the result is empty or NO_ANSWER
        if self.result == "" or NO_ANSWER in self.result:
            # if result is empty or NO_ANSWER, we should retry the query plan
            feedback = """
            There was no answer, which might mean there is a problem in your query.
            """
            suggested = "Retry the `query_plan` to try to get a non-null answer"
        else:
            feedback = msg.feedback
            suggested = msg.suggested_fix

        self.expecting_query_plan = True

        return f"""
        here is FEEDBACK about your QUERY PLAN, and a SUGGESTED FIX.
        Modify the QUERY PLAN if needed:
        ANSWER: {self.result}
        FEEDBACK: {feedback}
        SUGGESTED FIX: {suggested}
        """

    def answer_tool(self, msg: AnswerTool) -> QueryPlanAnswerTool:
        """Handle AnswerTool received from LanceRagAgent:
        Construct a QueryPlanAnswerTool with the answer"""
        self.result = msg.answer  # save answer to interpret feedback later
        assert self.curr_query_plan is not None
        query_plan_answer_tool = QueryPlanAnswerTool(
            plan=self.curr_query_plan,
            answer=msg.answer,
        )
        self.curr_query_plan = None  # reset
        return query_plan_answer_tool

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """
        Remind to use QueryPlanTool if we are expecting it.
        """
        if self.expecting_query_plan and self.n_query_plan_reminders < 5:
            self.n_query_plan_reminders += 1
            return """
            You FORGOT to use the `query_plan` tool/function, 
            OR you had a WRONG JSON SYNTAX when trying to use it.
            Re-try your response using the `query_plan` tool/function CORRECTLY.
            """
        self.n_query_plan_reminders = 0  # reset
        return None
