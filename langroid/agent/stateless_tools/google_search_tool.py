import os
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import Resource, build
from requests.models import Response

from langroid.agent.tool_message import ToolMessage


class GoogleSearchResult:
    """
    Class representing a Google Search result, containing the title, link,
    summary and full content of the result.
    """

    def __init__(
        self,
        title: str,
        link: str,
        max_content_length: int = 3500,
        max_summary_length: int = 300,
    ):
        """
        Args:
            title (str): The title of the search result.
            link (str): The link to the search result.
            max_content_length (int): The maximum length of the full content.
            max_summary_length (int): The maximum length of the summary.
        """
        self.title = title
        self.link = link
        self.max_content_length = max_content_length
        self.max_summary_length = max_summary_length
        self.full_content = self.get_full_content()
        self.summary = self.get_summary()

    def get_summary(self) -> str:
        return self.full_content[: self.max_summary_length]

    def get_full_content(self) -> str:
        response: Response = requests.get(self.link)
        soup: BeautifulSoup = BeautifulSoup(response.text, "lxml")
        text = " ".join(soup.stripped_strings)
        return text[: self.max_content_length]

    def __str__(self) -> str:
        return f"Title: {self.title}\nLink: {self.link}\nSummary: {self.summary}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "summary": self.summary,
            "full_content": self.full_content,
        }


class GoogleSearchTool(ToolMessage):
    request: str = "web_search"
    purpose: str = """
            To search the web and return up to <num_results> links relevant to 
            the given <query>. 
            """
    query: str
    num_results: int

    def handle(self) -> str:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        service: Resource = build("customsearch", "v1", developerKey=api_key)
        raw_results = (
            service.cse()
            .list(q=self.query, cx=cse_id, num=self.num_results)
            .execute()["items"]
        )

        search_results: List[GoogleSearchResult] = [
            GoogleSearchResult(result["title"], result["link"], 3500, 300)
            for result in raw_results
        ]
        # return Title and Link of each result, separated by two newlines
        return "\n\n".join(str(result) for result in search_results)
