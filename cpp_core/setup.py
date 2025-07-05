from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        "cpp_core",
        sources=[os.path.join("src", "core.cpp")],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language="c++",
        extra_compile_args=["-std=c++20"],
    )
]

setup(
    name="cpp_core",
    version="0.1",
    ext_modules=ext_modules,
    zip_safe=False,
)
