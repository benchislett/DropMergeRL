# cpp_core

This is the C++ core for the DropMergeRL project. It is designed for performance-critical operations and integrates seamlessly with the existing Python components.

## Features
- Creates a 7x5 grid of integers filled with zeros.
- Passes the grid as a NumPy ndarray to the Python caller.

## Requirements
- CMake >= 3.15
- Python >= 3.6
- pybind11

## Installation

To install the C++ core as an editable Python package, run:

```bash
pip install -e .
```

## Usage

```python
import cpp_core

grid = cpp_core.create_grid()
print(grid)
```
