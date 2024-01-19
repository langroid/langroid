---
draft: true
date: 2022-01-31
authors: 
  - pchalasani
categories:
  - test
  - blog
comments: true
---

# Test code snippets

```python
from langroid.language_models.base import LLMMessage, Role
msg = LLMMessage(
        content="What is the capital of Bangladesh?",
        role=Role.USER,
      )
```

<!-- more -->


# Test math notation

A nice equation is $e^{i\pi} + 1 = 0$, which is known as Euler's identity.
Here is a cool equation too, and in display mode:

$$
e = mc^2
$$

# Latex with newlines

Serious latex with `\\` for newlines renders fine:

$$
\begin{bmatrix}
a & b \\
c & d \\
e & f \\
\end{bmatrix}
$$

or a multi-line equation

$$
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
$$

<iframe src="https://langroid.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>


