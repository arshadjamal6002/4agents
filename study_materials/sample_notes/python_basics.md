# Python Basics

## Functions

Functions are first-class objects in Python: you can assign them,
pass them as arguments, and return them from other functions.

```python
def greet(name):
    return f"Hello, {name}"

say_hi = greet
print(say_hi("Ada"))
```

## Variables and Types

Python is dynamically typed. Names refer to objects; objects have types.
