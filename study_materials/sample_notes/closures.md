# Python Closures

## What is a closure?

A closure is a function that remembers variables from the enclosing scope
even after that scope has finished executing.

```python
def make_multiplier(n):
    def multiply(x):
        return x * n
    return multiply

double = make_multiplier(2)
print(double(5))  # 10
```

The inner function `multiply` closes over `n`.

## Common gotcha

Loop variables in closures are late-bound — they capture the variable,
not the value at definition time, unless you bind with a default argument.
