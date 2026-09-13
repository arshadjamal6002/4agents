# Python Decorators

## What is a decorator?

A decorator is a function that takes another function and extends its
behavior without permanently modifying it. Decorators are often written
using the `@` syntax.

```python
def shout(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return str(result).upper()
    return wrapper

@shout
def greet(name):
    return f"hello, {name}"

print(greet("ada"))  # HELLO, ADA
```

Decorators are closures that wrap callables.

## Common gotcha

Forgetting `functools.wraps` on the wrapper loses the original function's
name and docstring.
