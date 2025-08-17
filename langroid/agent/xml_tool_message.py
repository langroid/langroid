import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from lxml import etree
from pydantic import BaseModel, ConfigDict

from langroid.agent.tool_message import ToolMessage

# For Union type handling - check if we have Python 3.10+ UnionType
HAS_UNION_TYPE = False
try:
    from types import UnionType  # noqa: F401 # Used conditionally

    HAS_UNION_TYPE = True
except ImportError:
    pass


class XMLToolMessage(ToolMessage):
    """
    Abstract class for tools formatted using XML instead of JSON.

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

    _allow_llm_use: bool = True

    model_config = ConfigDict(
        # Inherit settings from ToolMessage
        extra="allow",
        arbitrary_types_allowed=False,
        validate_default=True,
        validate_assignment=True,
        json_schema_extra={"exclude": ["purpose", "id"]},
    )

    # XMLToolMessage-specific settings as class methods to avoid Pydantic
    # treating them as model fields
    @classmethod
    def _get_excluded_fields(cls) -> set[str]:
        return {"purpose", "id"}

    # Root element for XML formatting
    @classmethod
    def _get_root_element(cls) -> str:
        return "tool"

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
        # SECURITY: Initialize XMLParser with flags to prevent
        # XML External Entity (XXE), billion laughs, and external DTD attacks by
        # disabling entity resolution, DTD loading, and network access;
        # `strip_cdata=False` is needed to preserve
        # content within CDATA sections (e.g., for code).
        parser = etree.XMLParser(
            strip_cdata=False,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
        )
        root = etree.fromstring(formatted_string.encode("utf-8"), parser=parser)

        def parse_element(element: etree._Element) -> Any:
            # Skip elements starting with underscore
            if element.tag.startswith("_"):
                return {}

            field_info = cls.model_fields.get(element.tag)
            is_verbatim = (
                field_info
                and hasattr(field_info, "json_schema_extra")
                and field_info.json_schema_extra is not None
                and isinstance(field_info.json_schema_extra, dict)
                and field_info.json_schema_extra.get("verbatim", False)
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
                # For branch elements, handle potential lists or nested structures
                children = [parse_element(child) for child in element]
                if all(child.tag == element[0].tag for child in element):
                    # If all children have the same tag, treat as a list
                    return children
                else:
                    # Otherwise, treat as a dictionary
                    result = {child.tag: parse_element(child) for child in element}
                    # Check if this corresponds to a nested Pydantic model
                    if (
                        field_info
                        and isinstance(field_info.annotation, type)
                        and issubclass(field_info.annotation, BaseModel)
                    ):
                        return field_info.annotation(**result)
                    return result

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
        try:
            parsed_data = cls.extract_field_values(formatted_string)
            if parsed_data is None:
                return None

            # Use Pydantic's parse_obj to create and validate the instance
            return cls.model_validate(parsed_data)
        except Exception as e:
            from langroid.exceptions import XMLException

            raise XMLException(f"Error parsing XML: {str(e)}")

    @classmethod
    def find_verbatim_fields(
        cls, prefix: str = "", parent_cls: Optional[type[BaseModel]] = None
    ) -> List[str]:
        verbatim_fields = []
        for field_name, field_info in (parent_cls or cls).model_fields.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            if (
                hasattr(field_info, "json_schema_extra")
                and field_info.json_schema_extra is not None
                and isinstance(field_info.json_schema_extra, dict)
                and field_info.json_schema_extra.get("verbatim", False)
            ) or field_name == "code":
                verbatim_fields.append(full_name)
            if isinstance(field_info.annotation, type) and issubclass(
                field_info.annotation, BaseModel
            ):
                verbatim_fields.extend(
                    cls.find_verbatim_fields(full_name, field_info.annotation)
                )
        return verbatim_fields

    @classmethod
    def format_instructions(cls, tool: bool = False) -> str:
        fields = [
            f for f in cls.model_fields.keys() if f not in cls._get_excluded_fields()
        ]

        instructions = """
        To use this tool, please provide the required information in an XML-like 
        format. Here's how to structure your input:\n\n
        """

        preamble = "Placeholders:\n"
        xml_format = f"Formatting example:\n\n<{cls._get_root_element()}>\n"

        def format_field(
            field_name: str,
            field_type: Any,
            indent: str = "",
            path: str = "",
        ) -> None:
            nonlocal preamble, xml_format
            current_path = f"{path}.{field_name}" if path else field_name

            origin = get_origin(field_type)
            args = get_args(field_type)

            # Handle Union types (including Optional types like List[Person] | None)
            # Support both typing.Union and types.UnionType (Python 3.10+ | syntax)
            is_union = origin is Union
            if HAS_UNION_TYPE:
                from types import UnionType as _UnionType

                is_union = is_union or origin is _UnionType

            if is_union:
                # Filter out None type for Optional types
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    # This is an Optional type, process the non-None type
                    field_type = non_none_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                # If there are multiple non-None types, fall through to default handling

            if (
                origin is None
                and isinstance(field_type, type)
                and issubclass(field_type, BaseModel)
            ):
                preamble += (
                    f"{field_name.upper()} = [nested structure for {field_name}]\n"
                )
                xml_format += f"{indent}<{field_name}>\n"
                for sub_field, sub_field_info in field_type.model_fields.items():
                    format_field(
                        sub_field,
                        sub_field_info.annotation,
                        indent + "  ",
                        current_path,
                    )
                xml_format += f"{indent}</{field_name}>\n"
            elif origin in (list, List) or (field_type is list):
                item_type = args[0] if args else Any
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    preamble += (
                        f"{field_name.upper()} = "
                        f"[list of nested structures for {field_name}]\n"
                    )
                else:
                    preamble += (
                        f"{field_name.upper()} = "
                        f"[list of {getattr(item_type, '__name__', str(item_type))} "
                        f"for {field_name}]\n"
                    )
                xml_format += f"{indent}<{field_name}>\n"
                xml_format += (
                    f"{indent}  <item>"
                    f"[{getattr(item_type, '__name__', str(item_type))} value]"
                    f"</item>\n"
                )
                xml_format += f"{indent}  ...\n"
                xml_format += f"{indent}</{field_name}>\n"
            elif origin in (dict, Dict) or (
                isinstance(field_type, type) and issubclass(field_type, Mapping)
            ):
                key_type, value_type = args if len(args) == 2 else (Any, Any)
                preamble += (
                    f"{field_name.upper()} = "
                    f"[dictionary with "
                    f"{getattr(key_type, '__name__', str(key_type))} keys and "
                    f"{getattr(value_type, '__name__', str(value_type))} values]\n"
                )
                xml_format += f"{indent}<{field_name}>\n"
                xml_format += (
                    f"{indent}  <{getattr(key_type, '__name__', str(key_type))}>"
                    f"[{getattr(value_type, '__name__', str(value_type))} value]"
                    f"</{getattr(key_type, '__name__', str(key_type))}>\n"
                )
                xml_format += f"{indent}  ...\n"
                xml_format += f"{indent}</{field_name}>\n"
            else:
                preamble += f"{field_name.upper()} = [value for {field_name}]\n"
                if current_path in verbatim_fields:
                    xml_format += (
                        f"{indent}<{field_name}>"
                        f"<![CDATA[{{{field_name.upper()}}}]]></{field_name}>\n"
                    )
                else:
                    xml_format += (
                        f"{indent}<{field_name}>"
                        f"{{{field_name.upper()}}}</{field_name}>\n"
                    )

        verbatim_fields = cls.find_verbatim_fields()

        for field in fields:
            field_info = cls.model_fields[field]
            field_type = field_info.annotation
            # Ensure we have a valid type
            if field_type is None:
                continue
            format_field(field, field_type)

        xml_format += f"</{cls._get_root_element()}>"

        verbatim_alert = ""
        if len(verbatim_fields) > 0:
            verbatim_alert = f"""
            EXTREMELY IMPORTANT: For these fields:
            {', '.join(verbatim_fields)},
            the contents MUST be wrapped in a CDATA section, and the content
            must be written verbatim WITHOUT any modifications or escaping,
            such as spaces, tabs, indents, newlines, quotes, etc.
            """

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

        def create_element(
            parent: etree._Element, name: str, value: Any, path: str = ""
        ) -> None:
            if value is None:
                return

            elem = etree.SubElement(parent, name)
            current_path = f"{path}.{name}" if path else name

            if isinstance(value, list):
                for item in value:
                    create_element(elem, "item", item, current_path)
            elif isinstance(value, dict):
                for k, v in value.items():
                    create_element(elem, k, v, current_path)
            elif isinstance(value, BaseModel):
                # Handle nested Pydantic models
                for field_name, field_value in value.model_dump().items():
                    create_element(elem, field_name, field_value, current_path)
            else:
                if current_path in self.__class__.find_verbatim_fields():
                    elem.text = etree.CDATA(str(value))
                else:
                    elem.text = str(value)

        root = etree.Element(self._get_root_element())
        exclude_fields: set[str] = self._get_excluded_fields()
        for name, value in self.model_dump().items():
            if name not in exclude_fields:
                create_element(root, name, value)

        result = etree.tostring(root, encoding="unicode", pretty_print=True)
        if not isinstance(result, str):
            raise ValueError("Unexpected non-string result from etree.tostring")
        return result

    @classmethod
    def find_candidates(cls, text: str) -> List[str]:
        """
        Finds XML-like tool message candidates in text, with relaxed opening tag rules.

        Args:
            text: Input text to search for XML structures.

        Returns:
            List of XML strings. For fragments missing the root opening tag but having
            valid XML structure and root closing tag, prepends the root opening tag.

        Example:
            With root_tag="tool", given:
            "Hello <field1>data</field1> </tool>"
            Returns: ["<tool><field1>data</field1></tool>"]
        """

        root_tag = cls._get_root_element()
        opening_tag = f"<{root_tag}>"
        closing_tag = f"</{root_tag}>"

        candidates = []
        pos = 0
        while True:
            # Look for either proper opening tag or closing tag
            start_normal = text.find(opening_tag, pos)
            end = text.find(closing_tag, pos)

            if start_normal == -1 and end == -1:
                break

            if start_normal != -1:
                # Handle normal case (has opening tag)
                end = text.find(closing_tag, start_normal)
                if end != -1:
                    candidates.append(text[start_normal : end + len(closing_tag)])
                    pos = max(end + len(closing_tag), start_normal + 1)
                    continue
                elif start_normal == text.rfind(opening_tag):
                    # last fragment - ok to miss closing tag
                    candidates.append(text[start_normal:] + closing_tag)
                    return candidates
                else:
                    pos = start_normal + 1
                    continue

            if end != -1:
                # Look backwards for first XML tag
                text_before = text[pos:end]
                first_tag_match = re.search(r"<\w+>", text_before)
                if first_tag_match:
                    start = pos + first_tag_match.start()
                    candidates.append(
                        opening_tag + text[start : end + len(closing_tag)]
                    )
                pos = end + len(closing_tag)

        return candidates
