# Running tests with global settings

Sometimes it's useful to run tests with global settings, e.g.

```python
pytest -s tests/ --nc --show
```

The options `--nc` and `--show` are global settings that are defined in 
`tests/conftest.py` and can be used in any test file. See the file for more 
details. 