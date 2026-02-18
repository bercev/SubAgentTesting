#!/usr/bin/env python3

# Deep debugging to find the actual bug
from astropy.modeling import models as m
from astropy.modeling.separable import _separable, _coord_matrix, _cstack

print("=== Deep debugging of separability computation ===")

# Let's trace through exactly what happens in _separable for the problematic case
cm = m.Linear1D(10) & m.Linear1D(5)
nested_model = m.Pix2Sky_TAN() & cm

print("cm = Linear1D(10) & Linear1D(5)")
print("nested_model = Pix2Sky_TAN() & cm")

# The _separable function logic:
# 1. Check if transform._calculate_separability_matrix() is not NotImplemented
# 2. If not, return that
# 3. Else if isinstance(transform, CompoundModel):
#    - sepleft = _separable(transform.left)
#    - sepright = _separable(transform.right) 
#    - return _operators[transform.op](sepleft, sepright)
# 4. Else if isinstance(transform, Model):
#    - return _coord_matrix(transform, "left", transform.n_outputs)

print("\n=== Step by step tracing ===")

print("Step 1: _separable(nested_model)")
print(f"  nested_model is CompoundModel: {isinstance(nested_model, m.CompoundModel)}")
print(f"  nested_model.op = {nested_model.op}")

print("Step 2: _separable(nested_model.left)")
left_result = _separable(nested_model.left)
print(f"  _separable(Pix2Sky_TAN()) =")
print(left_result)

print("Step 3: _separable(nested_model.right)")
right_result = _separable(nested_model.right)
print(f"  _separable(cm) =")
print(right_result)

print("Step 4: _operators['&'](_separable(left), _separable(right))")
final_result = _cstack(left_result, right_result)
print(f"  _cstack result =")
print(final_result)

print("\n=== Now let's see what happens with the actual separability_matrix function ===")
from astropy.modeling.separable import separability_matrix

actual_result = separability_matrix(nested_model)
print(f"Actual separability_matrix result:")
print(actual_result)

print("\n=== Let's also test the individual components of cm ===")
print("cm.left = ", cm.left)
print("cm.right = ", cm.right)

print("\n=== Testing _coord_matrix directly on components ===")
try:
    # This should work for simple models
    coord_left = _coord_matrix(cm.left, "left", 2)
    print(f"_coord_matrix(cm.left, 'left', 2) =")
    print(coord_left)
except Exception as e:
    print(f"Error with cm.left: {e}")

try:
    coord_right = _coord_matrix(cm.right, "left", 2)
    print(f"_coord_matrix(cm.right, 'left', 2) =")
    print(coord_right)
except Exception as e:
    print(f"Error with cm.right: {e}")

# The key insight: when _separable is called on a compound model,
# it should recursively call _separable on left and right components
# and then combine them with _cstack

print("\n=== Let's manually trace the _separable logic for cm ===")
print("For cm = Linear1D(10) & Linear1D(5):")
print("  _separable(cm) should call:")
print("  sepleft = _separable(cm.left) = _separable(Linear1D(10))")
print("  sepright = _separable(cm.right) = _separable(Linear1D(5))")
print("  return _cstack(sepleft, sepright)")

sepleft_cm = _separable(cm.left)
sepright_cm = _separable(cm.right)
cstack_result = _cstack(sepleft_cm, sepright_cm)
print(f"Manual _cstack result for cm: {cstack_result}")

print("\n=== Let's also check if there's a difference in how the models are constructed ===")
# Let's see if there's a difference in the model expressions
print("cm expression:", cm)
print("cm.left expression:", cm.left)
print("cm.right expression:", cm.right)

# Let's also check if there's a difference in the _calculate_separability_matrix
print("\n=== Checking if models have custom separability ===")
print("cm._calculate_separability_matrix():", cm._calculate_separability_matrix())
print("cm.left._calculate_separability_matrix():", cm.left._calculate_separability_matrix())
print("cm.right._calculate_separability_matrix():", cm.right._calculate_separability_matrix())
print("Pix2Sky_TAN()._calculate_separability_matrix():", m.Pix2Sky_TAN()._calculate_separability_matrix())