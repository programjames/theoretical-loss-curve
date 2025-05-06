from sympy import divisors, mobius, prod, lambdify
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np

def subscript(n):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_digits)

def anyon(n):
    return prod((1-x**d)**-mobius(d) for d in divisors(n)).cancel()

def count(Z):
    return (x * Z.diff(x) / Z).cancel()

def loss(n, points):
    n = count(anyon(n))
    return np.exp(-lambdify(x, n)(points))

for n, t in [
    [1, np.linspace(1, 10, 10_000)],
    [2, np.linspace(-10, 10, 10_000)],
    [6, np.linspace(-2, 20, 10_000)],
]:
    plt.figure(figsize=(10, 6))
    plt.plot(t, loss(n, np.exp(t)))
    plt.title(f"ℤ{subscript(n)}")
    plt.xlabel("Training Iteration")
    plt.ylabel("Theoretical Loss")
    plt.savefig(f"theory_Z{n}.png", bbox_inches="tight")
