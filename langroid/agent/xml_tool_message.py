from typing import Any, Dict, List, Optional

from lxml import etree

from langroid.agent.tool_message import ToolMessage


class XMLToolMessage(ToolMessage):
    """
    Abstract class for tools formatted using XML instead of JSON.
    Mainly tested for non-nested tool structures.

    When a subclass defines a field with the attribute `verbatim=True`,
    instructions are sent to the LLM to ensure the field's content is:
        - preserved as is, including whitespace, indents, quotes, newlines, etc
            with no escaping, and
        - enclosed in a CDATA section in the XML output.
    This is useful for LLMs sending code as part of a tool;
    results can be far superior compared to sending code in JSON-formatted tools,
    where code needs to confirm to JSON's strict rules and escaping requirements.
    (see test_xml_tool_message.py for an example).

    """

    request: str
    purpose: str

    _allow_llm_use = True

    class Config(ToolMessage.Config):
        root_element = "tool"

    @classmethod
    def extract_field_values(cls, formatted_string: str) -> Optional[Dict[str, Any]]:
        """
        Extracts field values from an XML-formatted string.

        Args:
            formatted_string (str): The XML-formatted string to parse.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the extracted field
                values, where keys are the XML element names and values are their
                corresponding contents.
            Returns None if parsing fails or the root element is not a dictionary.

        Raises:
            etree.XMLSyntaxError: If the input string is not valid XML.
        """
        parser = etree.XMLParser(strip_cdata=False)
        root = etree.fromstring(formatted_string.encode("utf-8"), parser=parser)

        def parse_element(element: etree._Element) -> Any | Dict[str, Any] | str:
            # Skip elements starting with underscore
            field_info = cls.__fields__.get(element.tag)
            is_verbatim = field_info and field_info.field_info.extra.get(
                "verbatim", False
            )

            if is_verbatim:
                # For code elements, preserve the content as is, including whitespace
                content = element.text if element.text else ""
                # Strip leading and trailing triple backticks if present,
                # accounting for whitespace
                return (
                    content.strip().removeprefix("```").removesuffix("```").strip()
                    if content.strip().startswith("```")
                    and content.strip().endswith("```")
                    else content
                )
            elif len(element) == 0:
                # For non-code leaf elements, strip whitespace
                return element.text.strip() if element.text else ""
            else:
                # For branch elements, recurse
                return {child.tag: parse_element(child) for child in element}

        result = parse_element(root)
        if not isinstance(result, dict):
            return None
        # Filter out empty dictionaries from skipped underscore fields
        return {k: v for k, v in result.items() if v != {}}

    @classmethod
    def parse(cls, formatted_string: str) -> Optional["XMLToolMessage"]:
        """
        Parses the XML-formatted string and returns an instance of the class.

        Args:
            formatted_string (str): The XML-formatted string to parse.

        Returns:
            Optional["XMLToolMessage"]: An instance of the class if parsing succeeds,
                None otherwise.
        """
        parsed_data = cls.extract_field_values(formatted_string)
        if parsed_data is None:
            return None

        # Use Pydantic's parse_obj to create and validate the instance
        return cls.parse_obj(parsed_data)

    @classmethod
    def format_instructions(cls, tool: bool = False) -> str:
        """
        Instructions to the LLM showing how to use the XML tool.

        Args:
            tool: Not used in this implementation, kept for compatibility.

        Returns:
            str: instructions on how to use the XML message
        """
        fields = [
            f
            for f in cls.__fields__.keys()
            if f not in cls.Config.schema_extra.get("exclude", set())
        ]

        instructions = """
        To use this tool, please provide the required information in an XML-like 
        format. Here's how to structure your input:\n\n
        """

        preamble = "Placeholders:\n"
        for field in fields:
            preamble += f"{field.upper()} = [value for {field}]\n"

        verbatim_fields = [
            field
            for field, field_info in cls.__fields__.items()
            if field_info.field_info.extra.get("verbatim", False)
        ]

        verbatim_alert = ""
        if len(verbatim_fields) > 0:
            verbatim_alert = f"""
            EXTREMELY IMPORTANT: For these fields:
            {', '.join(verbatim_fields)},
            the contents MUST be wrapped in a CDATA section, and the content
            must be written verbatim WITHOUT any modifications or escaping,
            such as spaces, tabs, indents, newlines, quotes, etc.
            """
        xml_format = f"Formatting example:\n\n<{cls.Config.root_element}>\n"
        for field in fields:
            if field == "code":
                xml_format += f"  <{field}><![CDATA[{{{field.upper()}}}]]></{field}>\n"
            else:
                xml_format += f"  <{field}>{{{field.upper()}}}</{field}>\n"
        xml_format += f"</{cls.Config.root_element}>"

        examples_str = ""
        if cls.examples():
            examples_str = "EXAMPLES:\n" + cls.usage_examples()

        return f"""
            TOOL: {cls.default_value("request")}
            PURPOSE: {cls.default_value("purpose")} 

            {instructions}
            {preamble}
            {xml_format}

            Make sure to replace the placeholders with actual values 
            when using the tool.                
            {verbatim_alert}            
            {examples_str}
            """.lstrip()

    def format_example(self) -> str:
        """
        Format the current instance as an XML example.

        Returns:
            str: A string representation of the current instance in XML format.

        Raises:
            ValueError: If the result from etree.tostring is not a string.
        """
        root = etree.Element(self.Config.root_element)
        exclude_fields = self.Config.schema_extra.get("exclude", set())
        for name, value in self.dict().items():
            if name not in exclude_fields:
                elem = etree.SubElement(root, name)
                field_info = self.__class__.__fields__[name]
                is_verbatim = field_info.field_info.extra.get("verbatim", False)
                if is_verbatim:
                    elem.text = etree.CDATA(str(value))
                else:
                    elem.text = str(value)
        result = etree.tostring(root, encoding="unicode", pretty_print=True)
        if not isinstance(result, str):
            raise ValueError("Unexpected non-string result from etree.tostring")
        return result

    @classmethod
    def find_candidates(cls, text: str) -> List[str]:
        """
        Find and extract all potential XML tool messages from the given text.

        This method searches for XML-like structures in the input text that match
        the expected format of the tool message. It looks for opening and closing
        tags that correspond to the root element defined in the XMLToolMessage class,
        which is by default <tool>.

        Args:
            text (str): The input text to search for XML tool messages.

        Returns:
            List[str]: A list of strings, each representing a potential XML tool
                       message.
                       These candidates include both the opening and
                       closing tags, so that they are individually parseable.

        Note:
            This method ensures that all candidates are valid and parseable by
            inserting a closing tag if it's missing for the last candidate.
        """
        root_tag = cls.Config.root_element
        opening_tag = f"<{root_tag}>"
        closing_tag = f"</{root_tag}>"

        candidates = []
        start = 0
        while True:
            start = text.find(opening_tag, start)
            if start == -1:
                break
            end = text.find(closing_tag, start)
            if end == -1:
                # For the last candidate, insert the closing tag if it's missing
                candidate = text[start:]
                if not candidate.strip().endswith(closing_tag):
                    candidate += closing_tag
                candidates.append(candidate)
                break
            candidates.append(text[start : end + len(closing_tag)])
            start = end + len(closing_tag)

        return candidates
