#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

print("Testing separability matrix issue more carefully...")

# Simple case - should be diagonal
cm = m.Linear1D(10) & m.Linear1D(5)
print("cm = m.Linear1D(10) & m.Linear1D(5)")
print("separability_matrix(cm):")
result1 = separability_matrix(cm)
print(result1)
print("Expected: diagonal matrix")
print()

# More complex case - should be block diagonal
print("m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)")
result2 = separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
print(result2)
print("Expected: block diagonal - first 2x2 block for Pix2Sky_TAN, last 2x2 block for Linear1D")
print()

# Let's also test the individual components to understand the structure
print("m.Pix2Sky_TAN().n_inputs:", m.Pix2Sky_TAN().n_inputs)
print("m.Pix2Sky_TAN().n_outputs:", m.Pix2Sky_TAN().n_outputs)
print("m.Linear1D(10).n_inputs:", m.Linear1D(10).n_inputs)
print("m.Linear1D(10).n_outputs:", m.Linear1D(10).n_outputs)
print("m.Linear1D(5).n_inputs:", m.Linear1D(5).n_inputs)
print("m.Linear1D(5).n_outputs:", m.Linear1D(5).n_outputs)
print()

# Now let's see what happens with the nested case
print("Testing nested case more carefully:")
nested = m.Pix2Sky_TAN() & cm
print("nested = m.Pix2Sky_TAN() & cm")
print("nested.n_inputs:", nested.n_inputs)
print("nested.n_outputs:", nested.n_outputs)
print("separability_matrix(nested):")
result3 = separability_matrix(nested)
print(result3)
print()

# Let's also test the structure of cm to make sure it's what we expect
print("cm structure:")
print("cm.left:", cm.left)
print("cm.right:", cm.right)
print("cm.left.n_inputs:", cm.left.n_inputs)
print("cm.left.n_outputs:", cm.left.n_outputs)
print("cm.right.n_inputs:", cm.right.n_inputs)
print("cm.right.n_outputs:", cm.right.n_outputs)
print()

# Let's also test the individual components of the nested model
print("nested.left:", nested.left)
print("nested.right:", nested.right)
print("nested.left.n_inputs:", nested.left.n_inputs)
print("nested.left.n_outputs:", nested.left.n_outputs)
print("nested.right.n_inputs:", nested.right.n_inputs)
print("nested.right.n_outputs:", nested.right.n_outputs)