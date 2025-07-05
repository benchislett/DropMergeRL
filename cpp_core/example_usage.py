import numpy as np
import cpp_core

# Create a 7x5 grid
arr = cpp_core.create_grid().copy()
print("Initial grid:")
print(arr, type(arr), arr.dtype)

# Mutate in-place
cpp_core.increment_inplace(arr)
print("After increment_inplace:")
print(arr)

cpp_core.resolve_board_inplace(arr, 2, 10000)
print("After resolve_board_inplace:")
print(arr)

# Return a new array
doubled = cpp_core.double_array(arr)
print("Doubled array:")
print(doubled)

# Return a scalar
s = cpp_core.sum_array(arr)
print(f"Sum of elements: {s}")

# Wrap a numpy array in a C++ class
wrapper = cpp_core.GridWrapper(arr)
wrapper.add_scalar(10)
print("After GridWrapper.add_scalar(10):")
print(wrapper.get())

# Create a C++-owned array and expose as a class
owned = cpp_core.OwnedGrid(3, 4)
owned.set(1, 2, 42)
print("OwnedGrid as numpy:")
print(owned.as_numpy())
owned.as_numpy()[:] += 1
print(owned.as_numpy())
owned.as_numpy_mut()[:] += 1
print(owned.as_numpy())
print(f"OwnedGrid get(1,2): {owned.get(1,2)}")
