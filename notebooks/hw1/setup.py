from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "physics_engine",         # The name of the module you'll import in Python
        ["physics.cpp"],          # Path to your C++ source file
        cxx_std=11,               # C++ standard to use
    ),
]

setup(
    name="physics_engine",
    version="0.1",
    author="Darya Guettler",
    description="A high-performance C++ physics engine for ball simulations",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)