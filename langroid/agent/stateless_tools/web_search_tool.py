from googleapiclient.discovery import build

from langroid.agent.tool_message import ToolMessage

# Your API key and CSE ID here
api_key = "AIzaSyDeW2KKzS42IJ6Slxt2RNGyya8zHLupk9A"
cse_id = "a66afd8dc68444527"


class WebSearchTool(ToolMessage):
    request: str = "web_search"
    purpose: str = """
            To search the web and return up to <num_results> links relevant to 
            the given <query>. 
            """
    query: str
    num_results: int

    def handle(self) -> str:
        service = build("customsearch", "v1", developerKey=api_key)
        results = (
            service.cse()
            .list(q=self.query, cx=cse_id, num=self.num_results)
            .execute()["items"]
        )

        # return Title and Link of each result, separated by two newlines
        return "\n\n".join(res["title"] + "\n" + res["link"] for res in results)
