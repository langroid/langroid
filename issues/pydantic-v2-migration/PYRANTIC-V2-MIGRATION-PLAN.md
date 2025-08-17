# Pydantic v2 Migration Plan

## Executive Summary

This document outlines a systematic approach to migrate Langroid's internal codebase from using the `langroid.pydantic_v1` compatibility layer to native Pydantic v2, while maintaining complete backward compatibility for external users.

**Scope**: 89 files using `langroid.pydantic_v1` imports across the entire codebase
**Timeline**: 7 days (systematic phased approach)
**Risk**: Low (incremental migration with testing at each phase)

## Current State Analysis

### Pydantic Usage Statistics
- **Total files with pydantic_v1 imports**: 89
  - Core langroid modules: 41 files
  - Test files: 11 files  
  - Example files: 37 files
- **Current dependency**: `"pydantic<3.0.0,>=1"` (supports both v1 and v2)

### Key Patterns to Migrate

#### 1. Method Calls (75 total occurrences)
- `.dict()` → `.model_dump()` (39 occurrences)
- `.parse_obj()` → `.model_validate()` (9 occurrences)
- `.parse_raw()` → `.model_validate_json()` (2 occurrences)
- `.json()` → `.model_dump_json()` (4 occurrences)
- `.copy()` → `.model_copy()` (21 occurrences estimated)

#### 2. Configuration Classes (22 occurrences)
```python
# From:
class Config:
    extra = Extra.allow
    validate_assignment = True

# To:
model_config = ConfigDict(extra='allow', validate_assignment=True)
```

#### 3. Validators (2 occurrences)
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

#### 4. Import Patterns
```python
# From:
from langroid.pydantic_v1 import BaseModel, Field, BaseSettings

# To:
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
```

### High-Priority Files for Migration

#### Core Framework (Phase 2a)
1. `langroid/agent/base.py` - Base agent class
2. `langroid/agent/tool_message.py` - Tool message system
3. `langroid/agent/chat_agent.py` - Chat agent implementation
4. `langroid/agent/task.py` - Task execution system

#### Language Models (Phase 2b)
1. `langroid/language_models/openai_gpt.py` - OpenAI integration
2. `langroid/language_models/base.py` - Base LLM classes
3. `langroid/language_models/azure_openai.py` - Azure integration
4. Other LLM provider files (8 total)

#### Vector Stores (Phase 2c)
1. `langroid/vector_store/base.py` - Base vector store
2. `langroid/vector_store/qdrant.py` - Qdrant integration
3. `langroid/vector_store/chroma.py` - Chroma integration
4. Other vector store implementations (12 total)

## Migration Plan

### Phase 1: Infrastructure Setup (Day 1)

#### 1.1 Update Dependencies
- **File**: `pyproject.toml`
- **Changes**:
  ```toml
  # From:
  pydantic = "<3.0.0,>=1"
  
  # To:
  pydantic = "^2.0.0"
  pydantic-settings = "^2.0.0"
  ```

#### 1.2 Create Migration Scripts
- **Script 1**: `scripts/migrate_pydantic_imports.py` - Automated import replacement
- **Script 2**: `scripts/migrate_pydantic_methods.py` - Method call migration
- **Script 3**: `scripts/migrate_pydantic_configs.py` - Config class migration
- **Script 4**: `scripts/validate_migration.py` - Verification script

#### 1.3 Baseline Testing
- Run complete test suite: `pytest tests/`
- Document current test results
- Identify any existing Pydantic-related test failures

### Phase 2: Core Framework Migration (Days 2-4)

#### Phase 2a: Base Classes (Day 2)
**Files to migrate** (2 files):
1. `langroid/agent/base.py`
2. `langroid/agent/tool_message.py`

**Migration steps**:
1. Replace `langroid.pydantic_v1` imports with native Pydantic v2
2. Update `.dict()` calls to `.model_dump()`
3. Update `.parse_obj()` calls to `.model_validate()`
4. Convert Config classes to `model_config = ConfigDict()`
5. Run targeted tests: `pytest tests/main/test_agent.py tests/main/test_tool_message.py`

#### Phase 2b: Chat Agent Core (Day 3)
**Files to migrate** (2 files):
1. `langroid/agent/chat_agent.py`
2. `langroid/agent/task.py`

**Migration steps**:
1. Import migration
2. Method call updates (heavy `.dict()` usage in chat_agent.py)
3. Config class updates
4. Run targeted tests: `pytest tests/main/test_chat_agent.py tests/main/test_task.py`

#### Phase 2c: Language Models (Day 4a)
**Files to migrate** (8 files):
1. `langroid/language_models/openai_gpt.py` (highest priority)
2. `langroid/language_models/base.py`
3. `langroid/language_models/azure_openai.py`
4. Other LLM provider files

**Migration steps**:
1. Focus on `.parse_obj()` calls (common in LLM response parsing)
2. Update configuration classes
3. Run targeted tests: `pytest tests/main/test_llm.py`

#### Phase 2d: Vector Stores (Day 4b)
**Files to migrate** (12 files):
1. `langroid/vector_store/base.py`
2. `langroid/vector_store/qdrant.py`
3. `langroid/vector_store/chroma.py`
4. Other vector store implementations

**Migration steps**:
1. Heavy focus on `.dict()` calls (document serialization)
2. Update configuration patterns
3. Run targeted tests: `pytest tests/main/test_vector_store.py`

### Phase 3: Tests & Examples (Day 5)

#### Phase 3a: Test Files (Day 5a)
**Files to migrate** (11 files):
- All test files with `langroid.pydantic_v1` imports
- Focus on test utilities and fixtures

**Migration steps**:
1. Import migration
2. Update test assertion patterns
3. Run individual test files after migration

#### Phase 3b: Example Files (Day 5b)
**Files to migrate** (37 files):
- All example files in `examples/` directory
- Focus on quick-start examples first

**Migration steps**:
1. Import migration
2. Update example patterns
3. Run examples to verify functionality

### Phase 4: Compatibility Layer Update (Day 6)

#### 4.1 Update Compatibility Layer
**Files to modify**:
- `langroid/pydantic_v1/__init__.py`
- `langroid/pydantic_v1/main.py`

**Changes**:
```python
# Update to always import from Pydantic v2
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings
# Add deprecation warnings for external users
```

#### 4.2 Add Deprecation Warnings
- Add warnings for external users still importing from `langroid.pydantic_v1`
- Document migration path for external users

### Phase 5: Final Validation (Day 7)

#### 5.1 Comprehensive Testing
- Run complete test suite: `pytest tests/`
- Run with coverage: `pytest --cov=langroid tests/`
- Performance benchmarking comparison

#### 5.2 Verification Checklist
- [ ] All 89 files migrated from `langroid.pydantic_v1`
- [ ] Zero test failures
- [ ] All examples run successfully
- [ ] Backward compatibility maintained
- [ ] Performance improvements measurable
- [ ] Documentation updated

#### 5.3 Migration Verification Report
Create final report documenting:
- Files migrated and patterns updated
- Test results comparison
- Performance improvements
- Backward compatibility verification
- Any issues encountered and resolved

## Migration Patterns Reference

### Import Migrations
```python
# Before
from langroid.pydantic_v1 import BaseModel, Field, BaseSettings, ValidationError

# After
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings
```

### Method Call Migrations
```python
# Before
data = model.dict()
obj = Model.parse_obj(data)
json_str = model.json()
copy_obj = model.copy()

# After
data = model.model_dump()
obj = Model.model_validate(data)
json_str = model.model_dump_json()
copy_obj = model.model_copy()
```

### Config Class Migrations
```python
# Before
class MyModel(BaseModel):
    field: str
    
    class Config:
        extra = Extra.allow
        validate_assignment = True

# After
class MyModel(BaseModel):
    field: str
    
    model_config = ConfigDict(extra='allow', validate_assignment=True)
```

### Validator Migrations
```python
# Before
@validator('field')
def validate_field(cls, v):
    return v

# After
@field_validator('field')
@classmethod
def validate_field(cls, v):
    return v
```

## Risk Mitigation Strategies

### 1. Incremental Migration
- Migrate files in logical groups
- Test after each group
- Maintain rollback capability

### 2. Backward Compatibility
- Preserve all existing APIs
- No changes to public interfaces
- Compatibility layer remains functional

### 3. Comprehensive Testing
- Run tests after each migration phase
- Focus on integration tests
- Performance regression testing

### 4. Documentation
- Update migration status in real-time
- Document any breaking changes discovered
- Create troubleshooting guide

## Success Metrics

### Primary Metrics
- **Migration Coverage**: 100% of files migrated from `langroid.pydantic_v1`
- **Test Success Rate**: 100% of existing tests pass
- **Backward Compatibility**: Zero breaking changes for external users

### Secondary Metrics
- **Performance Improvement**: Measurable speed improvements
- **Memory Usage**: Reduced memory footprint
- **Code Quality**: Cleaner, more maintainable code

## Rollback Plan

If critical issues are discovered:
1. **Immediate**: Revert specific file changes
2. **Temporary**: Maintain both old and new patterns
3. **Final**: Complete rollback to compatibility layer only

## Post-Migration Tasks

### 1. Documentation Updates
- Update README with Pydantic v2 requirements
- Update contribution guidelines
- Create migration guide for external users

### 2. Future Cleanup
- Plan removal of compatibility layer (future version)
- Adopt Pydantic v2-only features
- Performance optimization opportunities

### 3. Communication
- Announce migration completion
- Provide migration support for users
- Update examples and tutorials

## Conclusion

This migration plan provides a systematic, low-risk approach to migrating Langroid from Pydantic v1 to v2. The phased approach ensures thorough testing at each stage while maintaining complete backward compatibility for external users.

The migration will unlock performance improvements, future-proof the codebase, and eliminate the maintenance burden of the compatibility layer while preserving all existing functionality.