import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from lxml import etree
from pydantic import BaseModel, ConfigDict, RootModel

from langroid.agent.tool_message import ToolMessage

# For Union type handling - check if we have Python 3.10+ UnionType
HAS_UNION_TYPE = False
try:
    from types import UnionType  # noqa: F401 # Used conditionally

    HAS_UNION_TYPE = True
except ImportError:
    pass


def _unwrap_optional(annotation: Any) -> Any:
    """Reduce `Optional[X]` (and `X | None`) to `X`, leaving anything else as is.

    `format_instructions` applies the same reduction when it decides how to
    show a field to the LLM, so parsing must apply it too: otherwise a field
    declared `Optional[List[str]]` is advertised as a list but parsed as a
    scalar.

    Args:
        annotation: A type annotation, possibly `None`.

    Returns:
        The single non-`None` member of an `Optional`/`Union` annotation, or
        the annotation unchanged when it is not such a union.
    """
    origin = get_origin(annotation)
    is_union = origin is Union
    if HAS_UNION_TYPE:
        from types import UnionType as _UnionType

        is_union = is_union or origin is _UnionType
    if not is_union:
        return annotation
    non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
    return non_none[0] if len(non_none) == 1 else annotation


def _is_model(annotation: Any) -> bool:
    """Whether `annotation` is a Pydantic model whose fields map onto tags.

    `RootModel` is excluded: its content is a single unnamed value, which may
    itself be a list, so its children are not named fields and the structural
    heuristics must be left to decide.
    """
    return (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and not issubclass(annotation, RootModel)
    )


def _child_field(annotation: Any, tag: str) -> Any:
    """`FieldInfo` of field `tag` when `annotation` is a model declaring it.

    Carries both the child's type and its metadata (notably `verbatim`), so a
    nested element is described by its own model's field rather than by
    whatever top-level field happens to share its tag.

    Args:
        annotation: The parent element's resolved annotation.
        tag: The child element's tag.

    Returns:
        The child's `FieldInfo`, or `None` if the parent is not a model
        declaring such a field.
    """
    if not _is_model(annotation):
        return None
    return annotation.model_fields.get(tag)


def _child_type(annotation: Any, is_mapping: bool) -> Any:
    """Annotation the children of an `annotation`-typed element should have.

    For a list that is its item type, for a mapping its value type; `None`
    when the element's type says nothing about its children (a model's
    children are resolved per-tag by `_field_type` instead).

    Args:
        annotation: The element's own (Optional-reduced) annotation.
        is_mapping: Whether the element is being parsed as a mapping.

    Returns:
        A type annotation for the children, or `None` if unknown.
    """
    args = get_args(annotation)
    if is_mapping:
        return args[1] if len(args) == 2 else None
    return args[0] if len(args) == 1 else None


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

        def parse_element(
            element: etree._Element, expected_field: Any = None, expected: Any = None
        ) -> Any:
            # Skip elements starting with underscore
            if element.tag.startswith("_"):
                return {}

            # Resolve what this element IS by position, not by tag name. Only a
            # direct child of the root is a top-level field, so only it may be
            # matched to one by name; a deeper element is described solely by
            # what its parent passes down (`expected_field` when the parent is
            # a model and so has field metadata, `expected` for a list's items
            # or a mapping's values). Looking deeper elements up by tag would
            # let a nested tag -- or a dict key -- that happens to collide with
            # a top-level field name borrow that field's type or its
            # `verbatim` flag.
            is_top_level = element is not root and element.getparent() is root
            annotation: Any
            if element is root:
                annotation = None
            elif is_top_level:
                top_field = cls.model_fields.get(element.tag)
                annotation = _unwrap_optional(
                    top_field.annotation if top_field is not None else None
                )
            else:
                annotation = _unwrap_optional(
                    expected_field.annotation
                    if expected_field is not None
                    else expected
                )
            origin = get_origin(annotation)
            is_list = origin is list or annotation is list
            is_dict = origin is dict or (
                isinstance(annotation, type) and issubclass(annotation, Mapping)
            )
            # `verbatim` stays a top-level-only flag, as before: nested
            # elements were never verbatim unless their tag happened to match a
            # top-level verbatim field, which is exactly the collision above.
            verbatim_field = cls.model_fields.get(element.tag) if is_top_level else None
            is_verbatim = (
                verbatim_field
                and hasattr(verbatim_field, "json_schema_extra")
                and verbatim_field.json_schema_extra is not None
                and isinstance(verbatim_field.json_schema_extra, dict)
                and verbatim_field.json_schema_extra.get("verbatim", False)
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
                # An empty container element (e.g. `<tags/>`) is an empty
                # collection, not an empty string: take the type from the
                # field's declared annotation rather than from the XML.
                if element is root:
                    return {}
                if is_list:
                    return []
                if is_dict:
                    return {}
                # For non-code leaf elements, strip whitespace
                return element.text.strip() if element.text else ""
            else:
                # For branch elements, handle potential lists or nested structures
                # The "all children share a tag" heuristic below misreads a
                # single-entry dict or a single-field nested model as a list,
                # so decide by declared type first where we have one.
                is_mapping = element is root or is_dict or _is_model(annotation)
                # Tell the children what they are expected to be, since they
                # cannot be looked up by tag: a list's items, a mapping's
                # values, or a nested model's same-named field.
                child_type = _child_type(annotation, is_mapping)
                if not is_mapping and all(
                    child.tag == element[0].tag for child in element
                ):
                    # If all children have the same tag, treat as a list
                    return [parse_element(child, None, child_type) for child in element]
                else:
                    # Otherwise, treat as a dictionary
                    result = {
                        child.tag: parse_element(
                            child, _child_field(annotation, child.tag), child_type
                        )
                        for child in element
                    }
                    # A nested model is left as a plain dict for the caller's
                    # `model_validate` to coerce. Building it here would run
                    # its validation before the containing tool's own
                    # `mode="before"` validators get to transform the value,
                    # and every caller validates the result anyway.
                    return result

        result = parse_element(root)
        if not isinstance(result, dict):
            return None
        # Drop only the skipped underscore fields; an empty dict can now be a
        # legitimate argument value, so it must not be filtered out by value.
        return {k: v for k, v in result.items() if not k.startswith("_")}

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
