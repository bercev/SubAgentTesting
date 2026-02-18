#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

print("Testing separability matrix issue...")

# Simple case - should be diagonal
cm = m.Linear1D(10) & m.Linear1D(5)
print("cm = m.Linear1D(10) & m.Linear1D(5)")
print("separability_matrix(cm):")
result1 = separability_matrix(cm)
print(result1)
print()

# More complex case - should be block diagonal
print("m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)")
result2 = separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
print(result2)
print()

# Nested case - this is where the bug occurs
print("m.Pix2Sky_TAN() & cm")
result3 = separability_matrix(m.Pix2Sky_TAN() & cm)
print(result3)
print()

# Let's also check the shapes
print("cm.n_inputs:", cm.n_inputs)
print("cm.n_outputs:", cm.n_outputs)
print("m.Pix2Sky_TAN().n_inputs:", m.Pix2Sky_TAN().n_inputs)
print("m.Pix2Sky_TAN().n_outputs:", m.Pix2Sky_TAN().n_outputs)
print("m.Pix2Sky_TAN() & cm.n_inputs:", (m.Pix2Sky_TAN() & cm).n_inputs)
print("m.Pix2Sky_TAN() & cm.n_outputs:", (m.Pix2Sky_TAN() & cm).n_outputs)