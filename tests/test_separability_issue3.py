#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

print("Testing the exact issue described...")

# Let's recreate the exact scenario from the issue description
cm = m.Linear1D(10) & m.Linear1D(5)

print("cm = m.Linear1D(10) & m.Linear1D(5)")
print("separability_matrix(cm):")
result1 = separability_matrix(cm)
print(result1)
print()

# This should be a diagonal matrix [[True, False], [False, True]]
# The issue is that when we nest this with Pix2Sky_TAN(), 
# the separability matrix should still be block diagonal but it's not

print("m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)")
result2 = separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
print(result2)
print()

# The problem case - nested compound model
print("m.Pix2Sky_TAN() & cm")
result3 = separability_matrix(m.Pix2Sky_TAN() & cm)
print(result3)
print()

# Let's analyze what should happen:
# - Pix2Sky_TAN() has 2 inputs and 2 outputs
# - cm (Linear1D(10) & Linear1D(5)) has 2 inputs and 2 outputs  
# - When we do Pix2Sky_TAN() & cm, we should get a 4x4 matrix
# - The first 2x2 block should be for Pix2Sky_TAN() 
# - The last 2x2 block should be for cm
# - The off-diagonal blocks should be all False (independent)

print("Expected structure for m.Pix2Sky_TAN() & cm:")
print("Should be block diagonal:")
print("[[True, True, False, False],")
print(" [True, True, False, False],")
print(" [False, False, True, False],")
print(" [False, False, False, True]]")
print()

# Let's also test if we can understand what's happening by looking at the individual components
print("Let's see what happens with a simpler nested case:")
simple_nested = m.Pix2Sky_TAN() & (m.Linear1D(10) & m.Linear1D(5))
print("simple_nested = m.Pix2Sky_TAN() & (m.Linear1D(10) & m.Linear1D(5))")
print("separability_matrix(simple_nested):")
result4 = separability_matrix(simple_nested)
print(result4)
print()

# Let's also test if the issue is with the specific way the models are constructed
print("Testing with explicit model construction:")
cm2 = m.Linear1D(10) & m.Linear1D(5)
print("cm2 = m.Linear1D(10) & m.Linear1D(5)")
print("cm2.n_inputs:", cm2.n_inputs)
print("cm2.n_outputs:", cm2.n_outputs)
print("separability_matrix(cm2):")
result5 = separability_matrix(cm2)
print(result5)
print()

# Now let's see what happens when we do the nested operation
nested2 = m.Pix2Sky_TAN() & cm2
print("nested2 = m.Pix2Sky_TAN() & cm2")
print("nested2.n_inputs:", nested2.n_inputs)
print("nested2.n_outputs:", nested2.n_outputs)
print("separability_matrix(nested2):")
result6 = separability_matrix(nested2)
print(result6)
print()

# Let's also check the structure of the nested model
print("nested2.left:", nested2.left)
print("nested2.right:", nested2.right)
print("nested2.left.n_inputs:", nested2.left.n_inputs)
print("nested2.left.n_outputs:", nested2.left.n_outputs)
print("nested2.right.n_inputs:", nested2.right.n_inputs)
print("nested2.right.n_outputs:", nested2.right.n_outputs)