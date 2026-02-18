#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

print("Let's understand the issue better by comparing different approaches...")

# Create the compound model as described in the issue
cm = m.Linear1D(10) & m.Linear1D(5)

print("cm = m.Linear1D(10) & m.Linear1D(5)")
print("cm.n_inputs:", cm.n_inputs)
print("cm.n_outputs:", cm.n_outputs)
print("separability_matrix(cm):")
result1 = separability_matrix(cm)
print(result1)
print("is_separable(cm):")
print(is_separable(cm))
print()

# Now let's see what happens with explicit construction
print("m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)")
model1 = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
print("model1.n_inputs:", model1.n_inputs)
print("model1.n_outputs:", model1.n_outputs)
print("separability_matrix(model1):")
result2 = separability_matrix(model1)
print(result2)
print("is_separable(model1):")
print(is_separable(model1))
print()

# Now the nested case
print("m.Pix2Sky_TAN() & cm")
model2 = m.Pix2Sky_TAN() & cm
print("model2.n_inputs:", model2.n_inputs)
print("model2.n_outputs:", model2.n_outputs)
print("separability_matrix(model2):")
result3 = separability_matrix(model2)
print(result3)
print("is_separable(model2):")
print(is_separable(model2))
print()

# Let's also check if there's a difference in the structure
print("Let's see the structure of the models:")
print("model1.left:", model1.left)
print("model1.right:", model1.right)
print("model2.left:", model2.left)
print("model2.right:", model2.right)
print()

# Let's also test if the issue is about the order of operations
print("Testing with explicit parentheses:")
model3 = (m.Pix2Sky_TAN() & m.Linear1D(10)) & m.Linear1D(5)
print("model3 = (m.Pix2Sky_TAN() & m.Linear1D(10)) & m.Linear1D(5)")
print("model3.n_inputs:", model3.n_inputs)
print("model3.n_outputs:", model3.n_outputs)
print("separability_matrix(model3):")
result4 = separability_matrix(model3)
print(result4)
print("is_separable(model3):")
print(is_separable(model3))
print()

# Let's also test with a different approach to see if there's a difference
print("Testing with explicit model construction:")
model4 = m.Pix2Sky_TAN() & (m.Linear1D(10) & m.Linear1D(5))
print("model4 = m.Pix2Sky_TAN() & (m.Linear1D(10) & m.Linear1D(5))")
print("model4.n_inputs:", model4.n_inputs)
print("model4.n_outputs:", model4.n_outputs)
print("separability_matrix(model4):")
result5 = separability_matrix(model4)
print(result5)
print("is_separable(model4):")
print(is_separable(model4))
print()

# Let's also check if the issue is with the specific models being used
print("Let's test with simpler models to see if we can reproduce the issue:")
simple_cm = m.Shift(1) & m.Shift(2)
print("simple_cm = m.Shift(1) & m.Shift(2)")
print("separability_matrix(simple_cm):")
result6 = separability_matrix(simple_cm)
print(result6)
print()

print("m.Pix2Sky_TAN() & simple_cm")
simple_nested = m.Pix2Sky_TAN() & simple_cm
print("separability_matrix(simple_nested):")
result7 = separability_matrix(simple_nested)
print(result7)
print()

# Let's also check the individual components of the nested model
print("simple_nested.left:", simple_nested.left)
print("simple_nested.right:", simple_nested.right)
print("simple_nested.left.n_inputs:", simple_nested.left.n_inputs)
print("simple_nested.left.n_outputs:", simple_nested.left.n_outputs)
print("simple_nested.right.n_inputs:", simple_nested.right.n_inputs)
print("simple_nested.right.n_outputs:", simple_nested.right.n_outputs)