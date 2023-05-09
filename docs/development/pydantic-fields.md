When defining a Pydantic class with `Field(.., title, description, etc.)`, you
can provide additional metadata for your class attributes, which can be helpful
for various purposes, such as documentation, validation, and serialization.

Here's a list of some benefits of using `Field`:

1. **Documentation**: Adding `title` and `description` makes your code more
   self-explanatory, serving as inline documentation for your class attributes.
   This can be especially helpful for complex or large codebases.

2. **Validation**: You can use the `Field` function to add validation
   constraints to your attributes, such
   as `min_length`, `max_length`, `gt`, `lt`, `ge`, `le`, etc. This can help
   ensure that the input data meets specific requirements.

3. **Default values**: You can set default values for your attributes using
   the `Field` function. This can be useful when you want to provide a default
   value for an attribute that also requires validation or additional metadata.

4. **Serialization customization**: The `Field` function allows you to customize
   the serialization process for your attributes using the `alias` parameter.
   This can be helpful when your serialized data needs to follow a specific
   naming convention or if you want to decouple your internal attribute names
   from the external representation.

5. **Integration with OpenAPI and JSON Schema**: If you're using FastAPI or
   other frameworks that leverage Pydantic, the metadata you provide
   with `Field` can be automatically used to generate OpenAPI documentation and
   JSON Schema definitions for your API. This can save you time and effort when
   maintaining API documentation.

Here's an example of a Pydantic class with `Field`:

```python
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(
        "John Doe",
        title="Full Name",
        description="The full name of the person, including first and last name."
    )
    age: int = Field(
        ...,  # or fill in actual default values
        title="Age",
        description="The age of the person in years.",
        gt=0,
        lt=150
    )
    address: str = Field(
        ...,
        title="Address",
        description="The home address of the person, including street, city, state/province, and country."
    )
```

In this example, the `Field` function is used to provide additional metadata for
each attribute in the `Person` class.

# Restricting possible values of a field

Using an `Enum` is an excellent way to restrict the possible values of a
field in a Pydantic class. By defining an enumeration, you can ensure that a
field only accepts specific values, making your code more robust and preventing
invalid input.

Here's an example of a Pydantic class with a field restricted to specific values
using an `Enum`:

```python
from enum import Enum
from pydantic import BaseModel


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"
    UNDISCLOSED = "undisclosed"


class Person(BaseModel):
    name: str
    age: int
    address: str
    gender: Gender


# This will work
valid_person = Person(name="John Doe", age=30,
                      address="123 Main St, Anytown, Anystate, USA",
                      gender=Gender.MALE)

# This will raise a validation error
invalid_person = Person(name="Jane Doe", age=25,
                        address="456 Elm St, Anytown, Anystate, USA",
                        gender="invalid-gender")
```

In this example, the `Gender` enumeration is created using Python's
built-in `enum` module. The `gender` field in the `Person` class is then set to
accept only the values defined in the `Gender` enumeration. When you try to
create a `Person` instance with an invalid gender, Pydantic will raise a
validation error.

# Using the string value of an Enum

When you define `class Gender(str, Enum)`, you are creating a subclass of
both `str` and `Enum`. This is known as "mixing-in" a data type, and it allows
the enumeration to inherit the behaviors of both classes.

In this case, by subclassing `str`, you allow the `Gender` enumeration to
inherit the string methods and behaviors. This can be useful for serialization,
as the enumeration will be serialized as a string directly.

Here's an example of a Pydantic class using `class Gender(str, Enum)`:

```python
from enum import Enum
from pydantic import BaseModel


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"
    UNDISCLOSED = "undisclosed"


class Person(BaseModel):
    name: str
    age: int
    address: str
    gender: Gender


person = Person(name="John Doe", age=30,
                address="123 Main St, Anytown, Anystate, USA",
                gender=Gender.MALE)

print(person.json())
```

The output will be:

```
{"name": "John Doe", "age": 30, "address": "123 Main St, 
 Anytown, Anystate, USA", "gender": "male"}
```

As you can see, the `gender` attribute is serialized as a string ("male")
directly. If you didn't subclass `str`, the output would look different:

```python
from enum import Enum
from pydantic import BaseModel


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"
    UNDISCLOSED = "undisclosed"


class Person(BaseModel):
    name: str
    age: int
    address: str
    gender: Gender


person = Person(name="John Doe", age=30,
                address="123 Main St, Anytown, Anystate, USA",
                gender=Gender.MALE)

print(person.json())
```

The output would be:

```
{"name": "John Doe", "age": 30, 
  "address": "123 Main St, Anytown, Anystate, USA", "gender": "Gender.MALE"}
```

By subclassing `str`, you ensure that the enumeration is serialized as a string,
which is often more convenient and easier to understand.

# Optional fields

To specify optional fields in a Pydantic class, you can use the `Optional` type
from the `typing` module. By default, Pydantic considers fields with a default
value of `None` to be optional as well. Here's an example of a Pydantic class
with optional fields:

```python
from typing import Optional
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    address: str
    phone_number: Optional[str] = None
    email: Optional[str] = None
```

In this example, both the `phone_number` and `email` fields are marked as
optional using the `Optional` type, and their default values are set to `None`.
When creating an instance of the `Person` class, you can choose to provide
values for these fields or leave them empty:

```python
# Without optional fields
person1 = Person(name="John Doe", age=30,
                 address="123 Main St, Anytown, Anystate, USA")

# With optional fields
person2 = Person(name="Jane Doe", age=25,
                 address="456 Elm St, Anytown, Anystate, USA",
                 phone_number="555-1234", email="jane.doe@example.com")
```

Both `person1` and `person2` instances are valid, even though `person1` doesn't
have values for the optional fields `phone_number` and `email`.

# Methods in a Pydantic class

A Pydantic class can have other methods, just like typical Python classes.
Pydantic classes inherit from the `BaseModel` class, which itself is a Python
class. You can add methods to your Pydantic class to implement custom behavior,
perform calculations, or manipulate the data in some way.

Here's an example of a Pydantic class with additional methods:

```python
from typing import Optional
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    address: str
    phone_number: Optional[str] = None
    email: Optional[str] = None

    def is_adult(self) -> bool:
        return self.age >= 18

    def get_email_domain(self) -> Optional[str]:
        if self.email:
            return self.email.split('@')[-1]
        return None


person = Person(name="John Doe", age=30,
                address="123 Main St, Anytown, Anystate, USA",
                email="john.doe@example.com")

print(person.is_adult())  # Output: True
print(person.get_email_domain())  # Output: 'example.com'
```

In this example, we have added two methods to the `Person` class: `is_adult()`
and `get_email_domain()`. The `is_adult()` method checks if the person's age is
greater than or equal to 18, and the `get_email_domain()` method returns the
email domain of the person if the email attribute is provided.