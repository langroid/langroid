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

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.lance_tools import (
    QueryPlan,
    QueryPlanAnswerTool,
    QueryPlanFeedbackTool,
    QueryPlanTool,
)
from langroid.utils.constants import DONE, NO_ANSWER, PASS_TO

logger = logging.getLogger(__name__)


class LanceQueryPlanAgentConfig(ChatAgentConfig):
    name: str = "LancePlanner"
    critic_name: str = "QueryPlanCritic"
    doc_agent_name: str = "LanceRAG"
    doc_schema: str = ""
    use_tools = False
    max_retries: int = 5  # max number of retries for query plan
    use_functions_api = True

    system_message = f"""
    You will receive a QUERY, to be answered based on an EXTREMELY LARGE collection
    of documents you DO NOT have access to, but your ASSISTANT does.
    You only know that these documents have a special `content` field
    and additional FILTERABLE fields in the SCHEMA below:  
    
    {{doc_schema}}
    
    Based on the QUERY and the above SCHEMA, your task is to determine a QUERY PLAN,
    consisting of:
    -  a FILTER (can be empty string) that would help the ASSISTANT to answer the query.
        Remember the FILTER can refer to ANY fields in the above SCHEMA
        EXCEPT the `content` field of the documents. 
        ONLY USE A FILTER IF EXPLICITLY MENTIONED IN THE QUERY.
        TO get good results, for STRING MATCHES, consider using LIKE instead of =, e.g.
        "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'"
    - a possibly REPHRASED QUERY to be answerable given the FILTER.
        Keep in mind that the ASSISTANT does NOT know anything about the FILTER fields,
        so the REPHRASED QUERY should NOT mention ANY FILTER fields.
        The answer will answer based on documents whose CONTENTS match the QUERY, 
        possibly REPHRASED. 
    - an OPTIONAL SINGLE-LINE Pandas-dataframe calculation/aggregation string 
        that can be used to calculate the answer to the original query, 
        e.g. "df["rating"].mean()",
        or "df.groupby("director").mean()["rating"]", 
        or EMPTY string if no calc is needed. 
        The dataframe calc CAN refer to the `content` field.
        If a DataFrame calculation is NOT needed, leave this field EMPTY.
        
        IMPORTANT: The DataFrame `df` in this calculation is the result of 
        applying the FILTER AND REPHRASED QUERY to the documents.
    
    
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
    
    The FILTER must be a SQL-like condition, e.g. 
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
            
    If you keep getting feedback or keep getting a {NO_ANSWER} from the assistant
    at least 3 times, then simply say '{DONE} {NO_ANSWER}' and nothing else.
      
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
        self.curr_query_plan: QueryPlan | None = None
        # how many times re-trying query plan in response to feedback:
        self.n_retries: int = 0
        self.result: str = ""  # answer received from LanceRAG
        # This agent should generate the QueryPlanTool
        # as well as handle it for validation
        self.enable_message(QueryPlanTool, use=True, handle=True)
        self.enable_message(QueryPlanFeedbackTool, use=False, handle=True)

    def query_plan(self, msg: QueryPlanTool) -> str:
        """Valid, forward to RAG Agent"""
        # save, to be used to assemble QueryPlanResultTool
        if len(msg.plan.dataframe_calc.split("\n")) > 1:
            return "DATAFRAME CALCULATION must be a SINGLE LINE; Retry the `query_plan`"
        self.curr_query_plan = msg.plan
        return PASS_TO + self.config.doc_agent_name

    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> str:
        """Process Critic feedback on QueryPlan + Answer from RAG Agent"""
        # We should have saved answer in self.result by this time,
        # since this Agent seeks feedback only after receiving RAG answer.
        if msg.suggested_fix == "":
            self.n_retries = 0
            # This means the Query Plan or Result is good, as judged by Critic
            if self.result == "":
                # This was feedback for query with no result
                return "QUERY PLAN LOOKS GOOD!"
            elif self.result == NO_ANSWER:
                return NO_ANSWER
            else:  # non-empty and non-null answer
                return DONE + " " + self.result
        self.n_retries += 1
        if self.n_retries >= self.config.max_retries:
            # bail out to avoid infinite loop
            self.n_retries = 0
            return DONE + " " + NO_ANSWER
        return f"""
        here is FEEDBACK about your QUERY PLAN, and a SUGGESTED FIX.
        Modify the QUERY PLAN if needed:
        FEEDBACK: {msg.feedback}
        SUGGESTED FIX: {msg.suggested_fix}
        """

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """
        Process answer received from RAG Agent:
         Construct a QueryPlanAnswerTool with the answer,
         and forward to Critic for feedback.
        """
        # TODO we don't need to use this fallback method. instead we can
        # first call result = super().agent_response(), and if result is None,
        # then we know there was no tool, so we run below code
        if (
            isinstance(msg, ChatDocument)
            and self.curr_query_plan is not None
            and msg.metadata.parent is not None
        ):
            # save result, to be used in query_plan_feedback()
            self.result = msg.content
            # assemble QueryPlanAnswerTool...
            query_plan_answer_tool = QueryPlanAnswerTool(
                plan=self.curr_query_plan,
                answer=self.result,
            )
            response_tmpl = self.agent_response_template()
            # ... add the QueryPlanAnswerTool to the response
            # (Notice how the Agent is directly sending a tool, not the LLM)
            response_tmpl.tool_messages = [query_plan_answer_tool]
            # set the recipient to the Critic so it can give feedback
            response_tmpl.metadata.recipient = self.config.critic_name
            self.curr_query_plan = None  # reset
            return response_tmpl
        if (
            isinstance(msg, ChatDocument)
            and not self.has_tool_message_attempt(msg)
            and msg.metadata.sender == lr.Entity.LLM
        ):
            # remind LLM to use the QueryPlanFeedbackTool
            return """
            You forgot to use the `query_plan` tool/function.
            Re-try your response using the `query_plan` tool/function.
            """
        return None
