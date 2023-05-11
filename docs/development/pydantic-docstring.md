# Docstrings for Pydantic Classes

To write Google-style docstrings for a Pydantic class with multiline comments
for each field, follow this structure:

1. Start with a brief description of the class.
2. Use the `Attributes:` keyword followed by a description of each attribute.
3. Add a blank line between each attribute description and use an indented block
   for multiline comments.

Here's an example of a Pydantic class with Google-style docstrings:

```python
from pydantic import BaseModel


class Person(BaseModel):
    """
    A class to represent a person.

    Attributes:
        name (str): The full name of the person.
            This should include both the first and last name.
            The name should not contain any special characters or numbers.

        age (int): The age of the person in years.
            The age must be a positive integer.
            Age should not exceed 150.

        address (str): The home address of the person.
            The address should include street, city, state/province, and country.
            It can also include an optional apartment or suite number.
    """

    name: str
    age: int
    address: str
```

In this example, each attribute is followed by a brief description and then a
more detailed explanation in the indented block. The comments provide guidance
on how the attributes should be used and any constraints or requirements.