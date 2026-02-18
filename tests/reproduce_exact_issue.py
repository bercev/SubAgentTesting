#!/usr/bin/env python3

# Exact reproduction of the issue as described
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

print("=== Exact reproduction of the issue ===")

# First case - simple compound model
cm = m.Linear1D(10) & m.Linear1D(5)
print("cm = m.Linear1D(10) & m.Linear1D(5)")
result1 = separability_matrix(cm)
print("separability_matrix(cm):")
print(result1)
print()

# Second case - complex compound model
result2 = separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
print("separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)):")
print(result2)
print()

# Third case - nested compound model (the problematic one)
result3 = separability_matrix(m.Pix2Sky_TAN() & cm)
print("separability_matrix(m.Pix2Sky_TAN() & cm):")
print(result3)
print()

print("=== Analysis ===")
print("Are result2 and result3 equal?", (result2 == result3).all())
print("Expected behavior: They should be equal because they represent the same mathematical operation")
print("But the user says the nested case shows 'inputs and outputs are no longer separable'")

# Let's also check if there's a difference in the actual model structure
print("\n=== Model structure analysis ===")
print("m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5) structure:")
model1 = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
print(f"  Expression: {model1}")
print(f"  n_inputs: {model1.n_inputs}")
print(f"  n_outputs: {model1.n_outputs}")

print("\nm.Pix2Sky_TAN() & cm structure:")
model2 = m.Pix2Sky_TAN() & cm
print(f"  Expression: {model2}")
print(f"  n_inputs: {model2.n_inputs}")
print(f"  n_outputs: {model2.n_outputs}")

# Let's also check if there's a difference in how the models are parsed
print("\n=== Checking if there's a difference in parsing ===")
print("Let's see if the models are actually equivalent:")
print(f"model1 == model2: {model1 == model2}")
print(f"model1 is model2: {model1 is model2}")

# Let's see if there's a difference in the _calculate_separability_matrix method
print("\n=== Checking _calculate_separability_matrix ===")
print(f"model1._calculate_separability_matrix(): {model1._calculate_separability_matrix()}")
print(f"model2._calculate_separability_matrix(): {model2._calculate_separability_matrix()}")

# Let's also test if there's a difference in the actual computation
print("\n=== Manual computation check ===")
from astropy.modeling.separable import _separable, _cstack

# For model1: Pix2Sky_TAN() & (Linear1D(10) & Linear1D(5))
# For model2: Pix2Sky_TAN() & cm

print("Computing _separable for components of model1:")
print("  _separable(Pix2Sky_TAN()):")
sep1_left = _separable(m.Pix2Sky_TAN())
print(sep1_left)

print("  _separable(Linear1D(10) & Linear1D(5)):")
sep1_right = _separable(m.Linear1D(10) & m.Linear1D(5))
print(sep1_right)

print("  _cstack result:")
manual1 = _cstack(sep1_left, sep1_right)
print(manual1)

print("\nComputing _separable for components of model2:")
print("  _separable(Pix2Sky_TAN()):")
sep2_left = _separable(m.Pix2Sky_TAN())
print(sep2_left)

print("  _separable(cm):")
sep2_right = _separable(cm)
print(sep2_right)

print("  _cstack result:")
manual2 = _cstack(sep2_left, sep2_right)
print(manual2)

print(f"\nManual results equal? {manual1.tolist() == manual2.tolist()}")