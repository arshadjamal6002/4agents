# Control Structures

Python uses indentation to define blocks for conditionals and loops.

## Conditionals

```python
n = 12
if n % 2 == 0 and n > 10:
    print("even and greater than 10")
elif n % 2 == 0:
    print("even but not > 10")
else:
    print("odd")
```

## Loops

```python
for i in range(3):
    print(i)

count = 0
while count < 3:
    print(count)
    count += 1
```

## Common mistakes

- Forgetting `:` after `if` / `for` / `while`
- Mixing tabs and spaces (IndentationError)
- Using `=` instead of `==` in conditions
