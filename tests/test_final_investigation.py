#!/usr/bin/env python3

from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

print("=== Final investigation of the separability issue ===")

# The user says:
# 1. separability_matrix(cm) where cm = m.Linear1D(10) & m.Linear1D(5) gives diagonal matrix
# 2. separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)) gives block diagonal matrix  
# 3. separability_matrix(m.Pix2Sky_TAN() & cm) gives the same as #2, but the user says "Suddenly the inputs and outputs are no longer separable?"

# Let me test if there's a difference in the actual behavior that's not obvious from the matrix output
# Maybe the issue is in the interpretation of what separable means

# Let's create a case where we can see a clear difference
print("Case 1: Simple compound model")
cm = m.Linear1D(10) & m.Linear1D(5)
print(f"cm = {cm}")
print(f"separability_matrix(cm):")
result1 = separability_matrix(cm)
print(result1)
print(f"is_separable(cm): {is_separable(cm)}")
print()

print("Case 2: Complex compound model")
complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
print(f"complex_model = {complex_model}")
print(f"separability_matrix(complex_model):")
result2 = separability_matrix(complex_model)
print(result2)
print(f"is_separable(complex_model): {is_separable(complex_model)}")
print()

print("Case 3: Nested compound model")
nested_model = m.Pix2Sky_TAN() & cm
print(f"nested_model = {nested_model}")
print(f"separability_matrix(nested_model):")
result3 = separability_matrix(nested_model)
print(result3)
print(f"is_separable(nested_model): {is_separable(nested_model)}")
print()

# The user says the nested case should be different, but they're not seeing it
# Let me check if there's a difference in the actual computation by looking at the _calculate_separability_matrix

print("=== Checking if _calculate_separability_matrix is being used ===")
print(f"cm._calculate_separability_matrix(): {cm._calculate_separability_matrix()}")
print(f"complex_model._calculate_separability_matrix(): {complex_model._calculate_separability_matrix()}")
print(f"nested_model._calculate_separability_matrix(): {nested_model._calculate_separability_matrix()}")

# Let's also check if there's a difference in how the models are structured
print("\n=== Model structure analysis ===")
print("cm structure:")
print(f"  type: {type(cm)}")
print(f"  op: {cm.op}")
print(f"  left: {cm.left}")
print(f"  right: {cm.right}")
print()

print("nested_model structure:")
print(f"  type: {type(nested_model)}")
print(f"  op: {nested_model.op}")
print(f"  left: {nested_model.left}")
print(f"  right: {nested_model.right}")
print()

# Let me check if there's a bug in the _separable function by looking at the actual implementation
# The issue might be in how the _cstack function handles the case when one of the components is already a matrix

print("=== Let's trace through what should happen ===")
print("For nested_model = Pix2Sky_TAN() & cm:")
print("  _separable(nested_model) should call:")
print("  _separable(nested_model.left) = _separable(Pix2Sky_TAN())")
print("  _separable(nested_model.right) = _separable(cm)")
print("  Then _operators['&'](_separable(Pix2Sky_TAN()), _separable(cm))")

# Let's see what happens when we manually call the functions
print("\n=== Manual tracing ===")
from astropy.modeling.separable import _separable, _cstack, _coord_matrix

# Get the separable matrices for components
print("Separable matrix for Pix2Sky_TAN():")
sep_pix2sky = _separable(m.Pix2Sky_TAN())
print(sep_pix2sky)

print("Separable matrix for cm:")
sep_cm = _separable(cm)
print(sep_cm)

print("Result of _cstack(sep_pix2sky, sep_cm):")
manual_result = _cstack(sep_pix2sky, sep_cm)
print(manual_result)

print("\n=== The key insight ===")
print("The user might be expecting that the nested case should show different separability")
print("But looking at the results, they're actually the same, which suggests the implementation")
print("is working correctly. Maybe the user misunderstood what the separability matrix means.")

# Let's try to understand what the user might be thinking by creating a case where there's a real difference
print("\n=== Creating a case that should show different behavior ===")

# What if we have a case where the nested structure would actually matter?
# Let's try a case where we have a model that's not separable
print("Let's test with a non-separable model to see the difference:")

# Create a model that's not separable
try:
    # This might not work, but let's see what happens
    from astropy.modeling import custom_model
    
    @custom_model
    def non_separable_model(x, y):
        return x + y  # This is not separable
    
    print("Testing with a non-separable model:")
    ns_model = non_separable_model()
    print(f"ns_model.separable: {ns_model.separable}")
    print(f"separability_matrix(ns_model):")
    print(separability_matrix(ns_model))
    
except Exception as e:
    print(f"Error creating custom model: {e}")

# Let's also check if there's a bug in the _coord_matrix function for compound models
print("\n=== Testing _coord_matrix with compound models ===")
try:
    # Test what _coord_matrix returns for a compound model
    cm_coord = _coord_matrix(cm, "left", 2)
    print(f"_coord_matrix(cm, 'left', 2):")
    print(cm_coord)
except Exception as e:
    print(f"Error in _coord_matrix: {e}")

print("\n=== Conclusion ===")
print("Looking at the results, the separability matrices for the nested and non-nested cases")
print("are identical, which suggests the implementation is working correctly.")
print("The user might have been expecting a different behavior that doesn't actually exist.")
print("Or there might be a subtle bug in the implementation that I haven't found yet.")