"""
QueryPlanCritic is a ChatAgent that is created with a specific document schema.

Its role is to provide feedback on a Query Plan, which consists of:
- filter condition if needed (or empty string if no filter is needed)
- query - a possibly rephrased query that can be used to match the `content` field
- dataframe_calc - a Pandas-dataframe calculation/aggregation string, possibly empty
- original_query - the original query for reference
- result - the answer received from an assistant that used this QUERY PLAN.

This agent has access to two tools:
- QueryPlanTool: The handler method for this tool re-writes the query plan
  in plain text (non-JSON) so the LLM can provide its feedback using the
  QueryPlanFeedbackTool.
- QueryPlanFeedbackTool: LLM uses this tool to provide feedback on the Query Plan
"""

import logging

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgentConfig,
)
from langroid.agent.special.lance_tools import (
    QueryPlanAnswerTool,
    QueryPlanFeedbackTool,
)
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.utils.constants import NO_ANSWER

logger = logging.getLogger(__name__)


class QueryPlanCriticConfig(LanceQueryPlanAgentConfig):
    name = "QueryPlanCritic"
    system_message = f"""
    You are an expert at carefully planning a query that needs to be answered
    based on a large collection of documents. These docs have a special `content` field
    and additional FILTERABLE fields in the SCHEMA below, along with the 
    SAMPLE VALUES for each field, and the DTYPE in PANDAS TERMINOLOGY.
    
    {{doc_schema}}
    
    The ORIGINAL QUERY is handled by a QUERY PLANNER who sends the PLAN to an ASSISTANT,
    who returns an ANSWER.
    
    You will receive a QUERY PLAN consisting of:
    - ORIGINAL QUERY from the user, which a QUERY PLANNER processes,
      to create a QUERY PLAN, to be handled by an ASSISTANT.
    - PANDAS-LIKE FILTER, WHICH CAN BE EMPTY (and it's fine if results sound reasonable)
      FILTER SHOULD ONLY BE USED IF EXPLICITLY REQUIRED BY THE QUERY.
      This filter selects the documents over which the REPHRASED QUERY will be applied,
      thus naturally, the Re-phrased Query should NOT mention any FILTER fields,
      since it applies to the documents AFTER FILTERING.
    - REPHRASED QUERY (CANNOT BE EMPTY) that will be used to match against the 
      CONTENT (not filterable) of the documents.
      In general the REPHRASED QUERY should be relied upon to match the CONTENT 
      of the docs. Thus the REPHRASED QUERY itself acts like a 
      SEMANTIC/LEXICAL/FUZZY FILTER since the Assistant is able to use it to match 
      the CONTENT of the docs in various ways (semantic, lexical, fuzzy, etc.). 
        Keep in mind that the ASSISTANT does NOT know anything about the FILTER fields,
        so the REPHRASED QUERY should NOT mention ANY FILTER fields.
        The assistant will answer based on documents whose CONTENTS match the QUERY, 
        possibly REPHRASED. 
        !!!!****THE REPHRASED QUERY SHOULD NEVER BE EMPTY****!!!
        
        
    - DATAFRAME CALCULATION, which must be a SINGLE LINE calculation (or empty),
        [NOTE ==> This calculation is applied AFTER the FILTER and REPHRASED QUERY.],
    - ANSWER received from an assistant that used this QUERY PLAN.
      IT IS TOTALLY FINE FOR THE ANSWER TO NOT MENTION ANY FILTERING CONDITIONS,
      or if the ANSWER STATEMENT is MISSING SOME CRITERIA in the ORIGINAL QUERY.

        Here is an example of a VALID Plan + Answer:
        
        ORIGINAL QUERY: "Which crime novels were written by Russian authors after 1900?"
        FILTER: "author_nationality == 'Russian' and year_written > 1900"
        REPHRASED QUERY: "crime novel" [NOTICE NO FILTER FIELDS MENTIONED!!!]
        DATAFRAME CALC: ""
        ANSWER: "The Master and Margarita by Mikhail Bulgakov" 
            [NOTICE the answer does NOT need to say "crime novel" or "russian author"]
            
            
        Other examples of VALID ANSWER for a given ORIGINAL QUERY:
        
        ORIGINAL QUERY: "Which mountain is taller than 8000 meters?"
        ANSWER: "Mount Everest" [NOTICE no mention of "taller than 8000 meters"]
        
        ORIGINAL QUERY: "Which country has hosted the most olympics?"
        ANSWER: "United States" [NOTICE no mention of "most olympics"]

    In addition to the above SCHEMA fields there is a `content` field which:
    - CANNOT appear in a FILTER, 
    - CAN appear in the DATAFRAME CALCULATION.
    THERE ARE NO OTHER FIELDS IN THE DOCUMENTS or in the RESULTING DATAFRAME.
        
    Your job is to act as a CRITIC and provide feedback, 
    ONLY using the `query_plan_feedback` tool, and DO NOT SAY ANYTHING ELSE.
    
    Here is how you must examine the QUERY PLAN + ANSWER:
    - ALL filtering conditions in the original query must be EXPLICITLY 
      mentioned in the FILTER, and the QUERY field should not be used for filtering.
    - If the ANSWER contains an ERROR message, then this means that the query
      plan execution FAILED, and your feedback should say INVALID along 
      with the ERROR message, `suggested_fix` that aims to help the assistant 
      fix the problem (or simply equals "address the the error shown in feedback")
    - Ask yourself, is the ANSWER in the expected form, e.g. 
        if the question is asking for the name of an ENTITY with max SIZE,
        then the answer should be the ENTITY name, NOT the SIZE!! 
    - If the ANSWER is in the expected form, then the QUERY PLAN is likely VALID,
      and your feedback should say VALID, with empty `suggested_fix`.
      ===> HOWEVER!!! Watch out for a spurious correct-looking answer, for EXAMPLE:
      the query was to find the ENTITY with a maximum SIZE, 
      but the dataframe calculation is find the SIZE, NOT the ENTITY!!      
    - If the ANSWER is {NO_ANSWER} or of the wrong form, 
      then try to DIAGNOSE the problem IN THE FOLLOWING ORDER:
      - DATAFRAME CALCULATION -- is it doing the right thing?
        Is it finding the Index of a row instead of the value in a column?
        Or another example: maybe it is finding the maximum population
           rather than the CITY with the maximum population?
        If you notice a problem with the DATAFRAME CALCULATION, then
        ONLY SUBMIT FEEDBACK ON THE DATAFRAME CALCULATION, and DO NOT
        SUGGEST ANYTHING ELSE.
      - If the DATAFRAME CALCULATION looks correct, then check if 
        the REPHRASED QUERY makes sense given the ORIGINAL QUERY and FILTER.
        If this is the problem, then ONLY SUBMIT FEEDBACK ON THE REPHRASED QUERY,
        and DO NOT SUGGEST ANYTHING ELSE.
      - If the REPHRASED QUERY looks correct, then check if the FILTER makes sense.
        REMEMBER: A filter should ONLY be used if EXPLICITLY REQUIRED BY THE QUERY.
     
     
     IMPORTANT!! The DATAFRAME CALCULATION is done AFTER applying the 
         FILTER and REPHRASED QUERY! Keep this in mind when evaluating 
         the correctness of the DATAFRAME CALCULATION.
    
    ALWAYS use `query_plan_feedback` tool/fn to present your feedback
    in the `feedback` field, and if any fix is suggested,
    present it in the `suggested_fix` field.
    DO NOT SAY ANYTHING ELSE OUTSIDE THE TOOL/FN.
    IF NO REVISION NEEDED, simply leave the `suggested_fix` field EMPTY,
    and SAY NOTHING ELSE
    and DO NOT EXPLAIN YOURSELF.        
    """


def plain_text_query_plan(msg: QueryPlanAnswerTool) -> str:
    plan = f"""
    OriginalQuery: {msg.plan.original_query}
    Filter: {msg.plan.filter}
    Rephrased Query: {msg.plan.query}
    DataframeCalc: {msg.plan.dataframe_calc}
    Answer: {msg.answer}
    """
    return plan


class QueryPlanCritic(ChatAgent):
    """
    Critic for LanceQueryPlanAgent, provides feedback on
    query plan + answer.
    """

    def __init__(self, cfg: LanceQueryPlanAgentConfig):
        super().__init__(cfg)
        self.config = cfg
        self.enable_message(QueryPlanAnswerTool, use=False, handle=True)
        self.enable_message(QueryPlanFeedbackTool, use=True, handle=True)
        self.enable_message(AgentDoneTool, use=False, handle=True)

    def init_state(self) -> None:
        super().init_state()
        self.expecting_feedback_tool = False

    def query_plan_answer(self, msg: QueryPlanAnswerTool) -> str:
        """Present query plan + answer in plain text (not JSON)
        so LLM can give feedback"""
        self.expecting_feedback_tool = True
        return plain_text_query_plan(msg)

    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> AgentDoneTool:
        """Format Valid so return to Query Planner"""
        self.expecting_feedback_tool = False
        # indicate this task is Done, and return the tool as result
        return AgentDoneTool(tools=[msg])

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Remind the LLM to use QueryPlanFeedbackTool since it forgot"""
        if self.expecting_feedback_tool:
            return """
            You forgot to use the `query_plan_feedback` tool/function.
            Re-try your response using the `query_plan_feedback` tool/function,
            remember to provide feedback in the `feedback` field,
            and if any fix is suggested, provide it in the `suggested_fix` field.
            """
        return None
