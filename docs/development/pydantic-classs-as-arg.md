Yes, you can use the class A itself as an argument to the register method of
class B without needing an instance of class A. You can achieve this by using
Python's first-class object support, which allows you to pass classes as
arguments to functions or methods. Here's an example to illustrate this:

```python
from pydantic import BaseModel


class A(BaseModel):
    pass


class B:
    def __init__(self):
        self.registered_messages = []

    def register(self, message_class):
        if message_class not in self.registered_messages:
            self.registered_messages.append(message_class)
            print(f"Registered {message_class.__name__}")

    def display_registered_messages(self):
        for msg in self.registered_messages:
            print(msg.__name__)


# Example usage
b = B()
b.register(A)
b.display_registered_messages()
```

In this example, the `register` method of class B accepts a class, not an
instance, as its argument. The method then checks if the provided class is not
already in the `registered_messages` list and registers it if necessary.
The `display_registered_messages` method simply prints the names of the
registered classes for demonstration purposes.

You can also add type hinting to make the code more clear and maintainable:

```python
from typing import Type


class B:
    def register(self, message_class: Type[A]):
# rest of the method
```