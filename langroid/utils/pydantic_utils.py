import logging
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Tuple,
    Type,
    get_args,
    get_origin,
    no_type_check,
)

from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger(__name__)


def has_field(model_class: Type[BaseModel], field_name: str) -> bool:
    """Check if a Pydantic model class has a field with the given name."""
    return field_name in model_class.__fields__


def _recursive_purge_dict_key(d: Dict[str, Any], k: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == k and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_dict_key(d[key], k)


@no_type_check
def _flatten_pydantic_model_ignore_defaults(
    model: Type[BaseModel],
    base_model: Type[BaseModel] = BaseModel,
) -> Type[BaseModel]:
    """
    Given a possibly nested Pydantic class, return a flattened version of it,
    by constructing top-level fields, whose names are formed from the path
    through the nested structure, separated by double underscores.

    This version ignores inherited defaults, so it is incomplete.
    But retaining it as it is simpler and may be useful in some cases.
    The full version is `flatten_pydantic_model`, see below.

    Args:
        model (Type[BaseModel]): The Pydantic model to flatten.
        base_model (Type[BaseModel], optional): The base model to use for the
            flattened model. Defaults to BaseModel.

    Returns:
        Type[BaseModel]: The flattened Pydantic model.
    """

    flattened_fields: Dict[str, Tuple[Any, ...]] = {}
    models_to_process = [(model, "")]

    while models_to_process:
        current_model, current_prefix = models_to_process.pop()

        for name, field in current_model.__annotations__.items():
            if issubclass(field, BaseModel):
                new_prefix = (
                    f"{current_prefix}{name}__" if current_prefix else f"{name}__"
                )
                models_to_process.append((field, new_prefix))
            else:
                flattened_name = f"{current_prefix}{name}"
                flattened_fields[flattened_name] = (field, ...)

    return create_model(
        "FlatModel",
        __base__=base_model,
        **flattened_fields,
    )


def flatten_pydantic_model(
    model: Type[BaseModel],
    base_model: Type[BaseModel] = BaseModel,
) -> Type[BaseModel]:
    """
    Given a possibly nested Pydantic class, return a flattened version of it,
    by constructing top-level fields, whose names are formed from the path
    through the nested structure, separated by double underscores.

    Args:
        model (Type[BaseModel]): The Pydantic model to flatten.
        base_model (Type[BaseModel], optional): The base model to use for the
            flattened model. Defaults to BaseModel.

    Returns:
        Type[BaseModel]: The flattened Pydantic model.
    """

    flattened_fields: Dict[str, Any] = {}
    models_to_process = [(model, "")]

    while models_to_process:
        current_model, current_prefix = models_to_process.pop()

        for name, field in current_model.__fields__.items():
            if isinstance(field.outer_type_, type) and issubclass(
                field.outer_type_, BaseModel
            ):
                new_prefix = (
                    f"{current_prefix}{name}__" if current_prefix else f"{name}__"
                )
                models_to_process.append((field.outer_type_, new_prefix))
            else:
                flattened_name = f"{current_prefix}{name}"

                if field.default_factory is not field.default_factory:
                    flattened_fields[flattened_name] = (
                        field.outer_type_,
                        field.default_factory,
                    )
                elif field.default is not field.default:
                    flattened_fields[flattened_name] = (
                        field.outer_type_,
                        field.default,
                    )
                else:
                    flattened_fields[flattened_name] = (field.outer_type_, ...)

    return create_model("FlatModel", __base__=base_model, **flattened_fields)


def flatten_pydantic_instance(
    instance: BaseModel,
    prefix: str = "",
    force_str: bool = False,
) -> Dict[str, Any]:
    """
    Given a possibly nested Pydantic instance, return a flattened version of it,
    as a dict where nested traversal paths are translated to keys a__b__c.

    Args:
        instance (BaseModel): The Pydantic instance to flatten.
        prefix (str, optional): The prefix to use for the top-level fields.
        force_str (bool, optional): Whether to force all values to be strings.

    Returns:
        Dict[str, Any]: The flattened dict.

    """
    flat_data: Dict[str, Any] = {}
    for name, value in instance.dict().items():
        # Assuming nested pydantic model will be a dict here
        if isinstance(value, dict):
            nested_flat_data = flatten_pydantic_instance(
                instance.__fields__[name].type_(**value),
                prefix=f"{prefix}{name}__",
                force_str=force_str,
            )
            flat_data.update(nested_flat_data)
        else:
            flat_data[f"{prefix}{name}"] = str(value) if force_str else value
    return flat_data


def nested_dict_from_flat(
    flat_data: Dict[str, Any],
    sub_dict: str = "",
) -> Dict[str, Any]:
    """
    Given a flattened version of a nested dict, reconstruct the nested dict.
    Field names in the flattened dict are assumed to be of the form
    "field1__field2__field3", going from top level down.

    Args:
        flat_data (Dict[str, Any]): The flattened dict.
        sub_dict (str, optional): The name of the sub-dict to extract from the
            flattened dict. Defaults to "" (extract the whole dict).

    Returns:
        Dict[str, Any]: The nested dict.

    """
    nested_data: Dict[str, Any] = {}
    for key, value in flat_data.items():
        if sub_dict != "" and not key.startswith(sub_dict + "__"):
            continue
        keys = key.split("__")
        d = nested_data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    if sub_dict != "":  # e.g. "payload"
        nested_data = nested_data[sub_dict]
    return nested_data


def pydantic_obj_from_flat_dict(
    flat_data: Dict[str, Any],
    model: Type[BaseModel],
    sub_dict: str = "",
) -> BaseModel:
    """Flattened dict with a__b__c style keys -> nested dict -> pydantic object"""
    nested_data = nested_dict_from_flat(flat_data, sub_dict)
    return model(**nested_data)


def clean_schema(model: Type[BaseModel], excludes: List[str] = []) -> Dict[str, Any]:
    """
    Generate a simple schema for a given Pydantic model,
    including inherited fields, with an option to exclude certain fields.
    Handles cases where fields are Lists or other generic types and includes
    field descriptions if available.

    Args:
        model (Type[BaseModel]): The Pydantic model class.
        excludes (List[str]): A list of field names to exclude.

    Returns:
        Dict[str, Any]: A dictionary representing the simple schema.
    """
    schema = {}

    for field_name, field_info in model.__fields__.items():
        if field_name in excludes:
            continue

        field_type = field_info.outer_type_
        description = field_info.field_info.description or ""

        # Handle generic types like List[...]
        if get_origin(field_type):
            inner_types = get_args(field_type)
            inner_type_names = [
                t.__name__ if hasattr(t, "__name__") else str(t) for t in inner_types
            ]
            field_type_str = (
                f"{get_origin(field_type).__name__}" f'[{", ".join(inner_type_names)}]'
            )
            schema[field_name] = {"type": field_type_str, "description": description}
        elif issubclass(field_type, BaseModel):
            # Directly use the nested model's schema,
            # integrating it into the current level
            nested_schema = clean_schema(field_type, excludes)
            schema[field_name] = {**nested_schema, "description": description}
        else:
            # For basic types, use 'type'
            schema[field_name] = {
                "type": field_type.__name__,
                "description": description,
            }

    return schema


@contextmanager
def temp_update(
    pydantic_object: BaseModel, updates: Dict[str, Any]
) -> Generator[None, None, None]:
    original_values = {}
    try:
        for field, value in updates.items():
            if hasattr(pydantic_object, field):
                # Save original value
                original_values[field] = getattr(pydantic_object, field)
                setattr(pydantic_object, field, value)
            else:
                # Raise error for non-existent field
                raise AttributeError(
                    f"The field '{field}' does not exist in the "
                    f"Pydantic model '{pydantic_object.__class__.__name__}'."
                )
        yield
    except ValidationError as e:
        # Handle validation error
        print(f"Validation error: {e}")
    finally:
        # Restore original values
        for field, value in original_values.items():
            setattr(pydantic_object, field, value)
