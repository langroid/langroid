# Pydantic v2 Migration Task Specification

## Current State

Langroid currently uses a compatibility layer at `langroid/pydantic_v1/` that:
- Imports from `pydantic.v1.*` when Pydantic v2 is installed
- Falls back to `pydantic.*` when Pydantic v1 is installed
- Allows the codebase to work with both Pydantic versions

This approach works but creates issues:
- Import ordering conflicts when users have Pydantic v2 in their projects
- Users cannot use Pydantic v2 features alongside Langroid
- Performance limitations (Pydantic v1 is slower than v2)
- Future maintenance burden

## Goal

Migrate Langroid's internal codebase to use Pydantic v2 directly while maintaining complete backward compatibility for external users.

## Specific Objectives

### 1. Replace Internal Imports
Replace all internal imports of `langroid.pydantic_v1` with direct imports from:
- `pydantic` (for BaseModel, Field, etc.)
- `pydantic_settings` (for BaseSettings)

### 2. Update Method Calls
Update all Pydantic v1 method patterns to v2 equivalents:
- `.dict()` → `.model_dump()`
- `.parse_obj()` → `.model_validate()`
- `.json()` → `.model_dump_json()`
- `.copy()` → `.model_copy()`
- `.__fields__` → `.model_fields`
- `.schema()` → `.model_json_schema()`
- And others as needed

### 3. Update Configuration Patterns
Replace Pydantic v1 config classes with v2 ConfigDict:
```python
# From:
class Config:
    extra = Extra.allow

# To:
model_config = ConfigDict(extra='allow')
```

### 4. Update Validators
Replace v1 validators with v2 field validators:
```python
# From:
@validator('field')
def validate_field(cls, v):
    return v

# To:
@field_validator('field')
@classmethod
def validate_field(cls, v):
    return v
```

### 5. Update Dependencies
Update `pyproject.toml` to require Pydantic v2:
```toml
pydantic = "^2.0.0"
pydantic-settings = "^2.0.0"
```

## Critical Requirements

### 1. Complete Backward Compatibility
- External users should experience ZERO breaking changes
- All existing APIs must continue to work
- No changes to public interfaces

### 2. No Feature Removal
- Every existing function, class, and module must be preserved
- No deletion of files, tests, or examples
- All functionality must remain intact

### 3. Comprehensive Coverage
Update ALL instances of Pydantic v1 usage in:
- Core langroid modules
- Tests
- Examples
- Documentation

## Success Criteria

1. **Zero Internal v1 Imports**: No `langroid.pydantic_v1` imports remain in internal code
2. **All Tests Pass**: Complete test suite passes without errors
3. **Backward Compatibility**: External users can upgrade without code changes
4. **Performance**: Benefits from Pydantic v2 performance improvements
5. **Future-Proof**: Codebase is ready for Pydantic v2-only features

## Implementation Approach

1. **Systematic Analysis**: Identify all files using Pydantic v1 patterns
2. **Priority-Based Migration**: Start with core files, then tests, then examples
3. **Pattern-Based Updates**: Apply consistent transformation patterns
4. **Incremental Testing**: Test after each phase to catch issues early
5. **Verification**: Comprehensive final testing and validation

## Compatibility Layer Strategy

The existing `langroid/pydantic_v1/` compatibility layer should be:
- **Preserved** for external users who might be importing from it
- **Updated** to import from Pydantic v2 instead of v1
- **Documented** as deprecated for future removal

## Testing Strategy

1. **Before Migration**: Run full test suite to establish baseline
2. **During Migration**: Run tests after each file group
3. **After Migration**: Comprehensive test suite validation
4. **Focus Areas**: Pay special attention to:
   - Tool message functionality
   - Agent operations
   - Configuration loading
   - Data serialization/deserialization

## Deliverables

1. **Updated Codebase**: All internal code using Pydantic v2
2. **Passing Tests**: Complete test suite passes
3. **Updated Dependencies**: pyproject.toml reflects Pydantic v2
4. **Documentation**: Migration notes and compatibility information
5. **Verification Report**: Confirmation of successful migration

## Timeline

This is a significant migration that should be approached systematically over several phases, with thorough testing at each stage to ensure no functionality is lost or broken.