#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

print("Investigating the separability matrix issue...")

# Let's understand what the user might be expecting
# The user says that when they do:
# cm = m.Linear1D(10) & m.Linear1D(5)
# separability_matrix(cm) gives diagonal matrix
# separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)) gives block diagonal
# But separability_matrix(m.Pix2Sky_TAN() & cm) gives the same as the previous case

# Let's see if there's a difference in the actual behavior that's not obvious from the matrix
# Let's trace through what should happen

print("=== Understanding the models ===")

# Simple case
cm = m.Linear1D(10) & m.Linear1D(5)
print(f"cm = Linear1D(10) & Linear1D(5)")
print(f"cm.left = {cm.left}")
print(f"cm.right = {cm.right}")
print(f"cm.left.n_inputs = {cm.left.n_inputs}, n_outputs = {cm.left.n_outputs}")
print(f"cm.right.n_inputs = {cm.right.n_inputs}, n_outputs = {cm.right.n_outputs}")
print()

# Complex case
complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
print(f"complex_model = Pix2Sky_TAN() & Linear1D(10) & Linear1D(5)")
print(f"complex_model.left = {complex_model.left}")
print(f"complex_model.right = {complex_model.right}")
print(f"complex_model.left.n_inputs = {complex_model.left.n_inputs}, n_outputs = {complex_model.left.n_outputs}")
print(f"complex_model.right.n_inputs = {complex_model.right.n_inputs}, n_outputs = {complex_model.right.n_outputs}")
print()

# Nested case
nested_model = m.Pix2Sky_TAN() & cm
print(f"nested_model = Pix2Sky_TAN() & cm")
print(f"nested_model.left = {nested_model.left}")
print(f"nested_model.right = {nested_model.right}")
print(f"nested_model.left.n_inputs = {nested_model.left.n_inputs}, n_outputs = {nested_model.left.n_outputs}")
print(f"nested_model.right.n_inputs = {nested_model.right.n_inputs}, n_outputs = {nested_model.right.n_outputs}")
print()

print("=== Checking separability matrices ===")
print("cm separability_matrix:")
result1 = separability_matrix(cm)
print(result1)
print("is_separable(cm):", is_separable(cm))
print()

print("complex_model separability_matrix:")
result2 = separability_matrix(complex_model)
print(result2)
print("is_separable(complex_model):", is_separable(complex_model))
print()

print("nested_model separability_matrix:")
result3 = separability_matrix(nested_model)
print(result3)
print("is_separable(nested_model):", is_separable(nested_model))
print()

# Let's also check if there's a difference in the _calculate_separability_matrix method
print("=== Checking if models have _calculate_separability_matrix ===")
print(f"cm has _calculate_separability_matrix: {hasattr(cm, '_calculate_separability_matrix')}")
print(f"complex_model has _calculate_separability_matrix: {hasattr(complex_model, '_calculate_separability_matrix')}")
print(f"nested_model has _calculate_separability_matrix: {hasattr(nested_model, '_calculate_separability_matrix')}")

# Let's see if we can understand what's happening by manually tracing through the logic
print("\n=== Manual trace of _separable logic ===")
print("For cm (Linear1D(10) & Linear1D(5)):")
print("  - It's a CompoundModel with op='&'")
print("  - _separable(cm.left) = _coord_matrix(Linear1D(10), 'left', 2)")
print("  - _separable(cm.right) = _coord_matrix(Linear1D(5), 'left', 2)")
print("  - _operators['&'](_separable(cm.left), _separable(cm.right)) = _cstack(...)")
print()

print("For nested_model (Pix2Sky_TAN() & cm):")
print("  - It's a CompoundModel with op='&'")
print("  - _separable(nested_model.left) = _coord_matrix(Pix2Sky_TAN(), 'left', 4)")
print("  - _separable(nested_model.right) = _separable(cm)")
print("  - _operators['&'](_separable(nested_model.left), _separable(nested_model.right)) = _cstack(...)")
print()

# Let's manually compute what we expect
print("=== Manual computation ===")
print("Let's manually compute what _coord_matrix should return for each component:")

# For Pix2Sky_TAN() with 4 outputs
pix2sky_coord = m.Pix2Sky_TAN()._coord_matrix(m.Pix2Sky_TAN(), "left", 4)
print("Pix2Sky_TAN() coord matrix (should be 4x2):")
print(pix2sky_coord)

# For Linear1D(10) with 2 outputs
linear10_coord = m.Linear1D(10)._coord_matrix(m.Linear1D(10), "left", 2)
print("Linear1D(10) coord matrix (should be 2x1):")
print(linear10_coord)

# For Linear1D(5) with 2 outputs  
linear5_coord = m.Linear1D(5)._coord_matrix(m.Linear1D(5), "left", 2)
print("Linear1D(5) coord matrix (should be 2x1):")
print(linear5_coord)

# Let's see if we can find the actual bug by looking at the _cstack function
print("\n=== Looking at _cstack function ===")
from astropy.modeling.separable import _cstack, _coord_matrix

# Test _cstack with simple components
print("Testing _cstack with simple models:")
simple_left = _coord_matrix(m.Pix2Sky_TAN(), "left", 4)
simple_right = _coord_matrix(cm, "right", 4)
print("Left matrix (Pix2Sky_TAN):")
print(simple_left)
print("Right matrix (cm):")
print(simple_right)
print("Result of _cstack:")
result_cstack = _cstack(simple_left, simple_right)
print(result_cstack)