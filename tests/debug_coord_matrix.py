#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import _coord_matrix, _cstack, _separable

print("=== Debugging _coord_matrix function ===")

# Let's trace through what happens with a compound model
cm = m.Linear1D(10) & m.Linear1D(5)

print("cm = Linear1D(10) & Linear1D(5)")
print(f"cm.left = {cm.left}")
print(f"cm.right = {cm.right}")
print(f"cm.n_outputs = {cm.n_outputs}")

# Test _coord_matrix on the compound model
print("\nTesting _coord_matrix on compound model:")
try:
    result = _coord_matrix(cm, "left", 2)
    print(f"_coord_matrix(cm, 'left', 2) =")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Test what _separable returns for cm
print("\nTesting _separable on cm:")
try:
    result = _separable(cm)
    print(f"_separable(cm) =")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Test what _separable returns for Pix2Sky_TAN()
print("\nTesting _separable on Pix2Sky_TAN():")
try:
    result = _separable(m.Pix2Sky_TAN())
    print(f"_separable(Pix2Sky_TAN()) =")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Test _cstack with these results
print("\nTesting _cstack:")
try:
    left_result = _separable(m.Pix2Sky_TAN())
    right_result = _separable(cm)
    final_result = _cstack(left_result, right_result)
    print(f"_cstack(_separable(Pix2Sky_TAN()), _separable(cm)) =")
    print(final_result)
except Exception as e:
    print(f"Error: {e}")

# Let's also test the actual separability matrix computation step by step
print("\n=== Full separability matrix computation ===")
print("For nested_model = Pix2Sky_TAN() & cm:")
print("1. _separable(nested_model.left) = _separable(Pix2Sky_TAN())")
print("2. _separable(nested_model.right) = _separable(cm)")
print("3. _operators['&'](_separable(Pix2Sky_TAN()), _separable(cm))")

# Let's see what the actual _operators mapping does
from astropy.modeling.separable import _operators
print(f"\nOperators mapping: {_operators}")

# Let's also check if there's a difference in how the models are structured
print("\n=== Model structure comparison ===")
nested_model = m.Pix2Sky_TAN() & cm
print(f"nested_model expression: {nested_model}")
print(f"nested_model.left: {nested_model.left}")
print(f"nested_model.right: {nested_model.right}")
print(f"nested_model.op: {nested_model.op}")

# Let's also test if there's a difference in how the models are parsed
print("\n=== Testing with explicit parentheses ===")
model1 = m.Pix2Sky_TAN() & (m.Linear1D(10) & m.Linear1D(5))
print(f"model1 = Pix2Sky_TAN() & (Linear1D(10) & Linear1D(5))")
print(f"model1 expression: {model1}")

model2 = (m.Pix2Sky_TAN() & m.Linear1D(10)) & m.Linear1D(5)
print(f"model2 = (Pix2Sky_TAN() & Linear1D(10)) & Linear1D(5)")
print(f"model2 expression: {model2}")

print("\n=== Comparing the actual matrices ===")
from astropy.modeling.separable import separability_matrix

result1 = separability_matrix(model1)
result2 = separability_matrix(model2)
print(f"model1 separability_matrix:\n{result1}")
print(f"model2 separability_matrix:\n{result2}")
print(f"Are they equal? {result1.tolist() == result2.tolist()}")