# Pydantic Abstract Classes

Pydantic is a library that allows you to define data models and validate data
using Python type annotations. To create an abstract base class (ABC) using
Pydantic, you can combine it with the built-in `abc` module. Here's an example
of how to create an abstract base class using Pydantic:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel


class AbstractAnimalModel(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True

    @abstractmethod
    def speak(self) -> str:
        pass


class DogModel(AbstractAnimalModel):
    name: str
    breed: str

    def speak(self) -> str:
        return f"{self.name} says: Woof!"


class CatModel(AbstractAnimalModel):
    name: str
    color: str

    def speak(self) -> str:
        return f"{self.name} says: Meow!"


if __name__ == "__main__":
    dog = DogModel(name="Buddy", breed="Golden Retriever")
    print(dog.speak())

    cat = CatModel(name="Whiskers", color="Gray")
    print(cat.speak())
```

In this example, we define an abstract base class `AbstractAnimalModel` that
inherits from both `ABC` and `BaseModel`. Inside the class, we define a `Config`
class to configure Pydantic's behavior. Finally, we define an abstract
method `speak` that will need to be implemented by subclasses.

We then create two subclasses, `DogModel` and `CatModel`, that inherit from
the `AbstractAnimalModel`. Each subclass defines its own set of fields (using
Pydantic's type annotations) and implements the `speak` method.

In the main block, we create instances of `DogModel` and `CatModel` and call
their `speak` methods to demonstrate the usage of the abstract base class.

# Role of the `Config` class

The `Config` class in this example serves to configure the behavior of the
Pydantic `BaseModel` for the specific model it's defined in, as well as any
subclasses of that model. In this case, it's defined in
the `AbstractAnimalModel` class. The purpose of the `Config` class here is to
set several options that influence the validation and assignment behavior of
Pydantic:

1. `arbitrary_types_allowed`: When set to `True`, this option allows Pydantic to
   accept any arbitrary types for the fields of the model, even if they're not
   explicitly defined in Pydantic. By default, this option is set to `False`,
   and Pydantic will only accept a limited set of types for fields.

2. `validate_all`: When set to `True`, this option ensures that all fields in
   the model are validated, even if they have default values. This can be useful
   to ensure that any default values provided meet the validation criteria. By
   default, this option is set to `False`, and only fields without default
   values are validated.

3. `validate_assignment`: When set to `True`, this option makes sure that any
   assignment to the fields of the model is validated during runtime. If the
   assigned value doesn't meet the validation criteria, a `ValidationError` will
   be raised. By default, this option is set to `False`, and no runtime
   validation is performed during assignment.

In the given example, the `Config` class is used to enable these three options,
allowing for more flexible and strict validation behavior. This configuration
will apply to both the `AbstractAnimalModel` and any subclasses that inherit
from it, such as `DogModel` and `CatModel`.