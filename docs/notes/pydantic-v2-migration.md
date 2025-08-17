# Pydantic v2 Migration Guide

## Overview

Langroid has fully migrated to Pydantic v2! All internal code now uses Pydantic v2 
patterns and imports directly from `pydantic`. This guide will help you update your 
code to work with the new version.

## Compatibility Layer (Deprecated)

If your code currently imports from `langroid.pydantic_v1`:

```python
# OLD - Deprecated
from langroid.pydantic_v1 import BaseModel, Field, BaseSettings
```

You'll see a deprecation warning. This compatibility layer now imports from Pydantic v2 
directly, so your code may continue to work, but you should update your imports:

```python
# NEW - Correct
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings  # Note: BaseSettings moved to pydantic_settings in v2
```

!!! note "BaseSettings Location Change"
    In Pydantic v2, `BaseSettings` has moved to a separate `pydantic_settings` package.
    You'll need to install it separately: `pip install pydantic-settings`

!!! warning "Compatibility Layer Removal"
    The `langroid.pydantic_v1` module will be removed in a future version. 
    Update your imports now to avoid breaking changes.

## Key Changes to Update

### 1. All Fields Must Have Type Annotations

!!! danger "Critical Change"
    In Pydantic v2, fields without type annotations are completely ignored!

```python
# WRONG - Fields without annotations are ignored in v2
class MyModel(BaseModel):
    name = "John"          # ❌ This field is IGNORED!
    age = 25               # ❌ This field is IGNORED!
    role: str = "user"     # ✅ This field works

# CORRECT - All fields must have type annotations
class MyModel(BaseModel):
    name: str = "John"     # ✅ Type annotation required
    age: int = 25          # ✅ Type annotation required
    role: str = "user"     # ✅ Already correct
```

This is one of the most common issues when migrating to v2. Always ensure every field has an explicit type annotation, even if it has a default value.

#### Special Case: Overriding Fields in Subclasses

!!! danger "Can Cause Errors"
    When overriding fields from parent classes without type annotations, you may get 
    actual errors, not just ignored fields!

This is particularly important when creating custom Langroid agent configurations:

```python
# WRONG - This can cause errors!
from langroid import ChatAgentConfig
from langroid.language_models import OpenAIGPTConfig

class MyAgentConfig(ChatAgentConfig):
    # ❌ ERROR: Missing type annotation when overriding parent field
    llm = OpenAIGPTConfig(chat_model="gpt-4")
    
    # ❌ ERROR: Even with Field, still needs type annotation
    system_message = Field(default="You are a helpful assistant")

# CORRECT - Always include type annotations when overriding
class MyAgentConfig(ChatAgentConfig):
    # ✅ Type annotation required when overriding
    llm: OpenAIGPTConfig = OpenAIGPTConfig(chat_model="gpt-4")
    
    # ✅ Type annotation with Field
    system_message: str = Field(default="You are a helpful assistant")
```

Without type annotations on overridden fields, you may see errors like:
- `ValueError: Field 'llm' requires a type annotation`
- `TypeError: Field definitions should be annotated`
- Validation errors when the model tries to use the parent's field definition

### 2. Stricter Type Validation for Optional Fields

!!! danger "Breaking Change"
    Pydantic v2 is much stricter about type validation. Fields that could accept `None` 
    in v1 now require explicit `Optional` type annotations.

```python
# WRONG - This worked in v1 but fails in v2
class CloudSettings(BaseSettings):
    private_key: str = None      # ❌ ValidationError: expects string, got None
    api_host: str = None         # ❌ ValidationError: expects string, got None

# CORRECT - Explicitly mark fields as optional
from typing import Optional

class CloudSettings(BaseSettings):
    private_key: Optional[str] = None    # ✅ Explicitly optional
    api_host: Optional[str] = None       # ✅ Explicitly optional
    
    # Or using Python 3.10+ union syntax
    client_email: str | None = None      # ✅ Also works
```

This commonly affects:
- Configuration classes using `BaseSettings`
- Fields with `None` as default value
- Environment variable loading where the var might not be set

If you see errors like:
```
ValidationError: Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

The fix is to add `Optional[]` or `| None` to the type annotation.

### 3. Model Serialization Methods

```python
# OLD (Pydantic v1)
data = model.dict()
json_str = model.json()
new_model = MyModel.parse_obj(data)
new_model = MyModel.parse_raw(json_str)

# NEW (Pydantic v2)
data = model.model_dump()
json_str = model.model_dump_json()
new_model = MyModel.model_validate(data)
new_model = MyModel.model_validate_json(json_str)
```

### 4. Model Configuration

```python
# OLD (Pydantic v1)
class MyModel(BaseModel):
    name: str
    
    class Config:
        extra = "forbid"
        validate_assignment = True

# NEW (Pydantic v2)
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    name: str
```

### 5. Field Validators

```python
# OLD (Pydantic v1)
from pydantic import validator

class MyModel(BaseModel):
    name: str
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v

# NEW (Pydantic v2)
from pydantic import field_validator

class MyModel(BaseModel):
    name: str
    
    @field_validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v
```

### 6. Custom Types and Validation

```python
# OLD (Pydantic v1)
from pydantic import parse_obj_as
from typing import List

data = [{"name": "Alice"}, {"name": "Bob"}]
users = parse_obj_as(List[User], data)

# NEW (Pydantic v2)
from pydantic import TypeAdapter
from typing import List

data = [{"name": "Alice"}, {"name": "Bob"}]
users = TypeAdapter(List[User]).validate_python(data)
```

## Common Patterns in Langroid

When working with Langroid's agents and tools:

### Tool Messages

```python
from pydantic import BaseModel, Field
from langroid.agent.tool_message import ToolMessage

class MyTool(ToolMessage):
    request: str = "my_tool"
    purpose: str = "Process some data"
    
    # Use Pydantic v2 patterns
    data: str = Field(..., description="The data to process")
    
    def handle(self) -> str:
        # Tool logic here
        return f"Processed: {self.data}"
```

### Agent Configuration

```python
from pydantic import ConfigDict
from langroid import ChatAgentConfig

class MyAgentConfig(ChatAgentConfig):
    model_config = ConfigDict(extra="forbid")
    
    custom_param: str = "default_value"
```

## Troubleshooting

### Import Errors

If you see `ImportError` or `AttributeError` after updating imports:
- Make sure you're using the correct v2 method names (e.g., `model_dump` not `dict`)
- Check that field validators use `@field_validator` not `@validator`
- Ensure `ConfigDict` is used instead of nested `Config` classes

### Validation Errors

Pydantic v2 has stricter validation in some cases:
- Empty strings are no longer coerced to `None` for optional fields
- Type coercion is more explicit
- Extra fields handling may be different

### Performance

Pydantic v2 is generally faster, but if you notice any performance issues:
- Use `model_validate` instead of creating models with `**dict` unpacking
- Consider using `model_construct` for trusted data (skips validation)

## Need Help?

If you encounter issues during migration:
1. Check the [official Pydantic v2 migration guide](https://docs.pydantic.dev/latest/migration/)
2. Review Langroid's example code for v2 patterns
3. Open an issue on the [Langroid GitHub repository](https://github.com/langroid/langroid/issues)