## C++ library for fast generating of normal and range images

We implemented the generation of depth and normal maps in C++. In order to call it from python, we are
using the [pybind11 library](https://github.com/pybind/pybind11). At least version 2.2 is required.

We recommended to use pip to install the library, e.g.

```
sudo -H pip3 install pybind11
```
(The package python3-pybind11 from the Ubuntu repositories maybe too old).

Our C++ code can be build with

```
cd src/map_building/c_utils
mkdir build && cd build
cmake ..
make
```

Note that depending on the setup of the pybind11 library, one has to give the path to the `.cmake` files
for the pybind library, e.g.:

```
cmake .. -Dpybind11_DIR=/usr/local/lib/python3.6/dist-packages/pybind11/share/cmake/pybind11
```
Or, one could add pybind11 as a subdirectory inside the c++ project and directly compile it. 
For more details we refer to the pybind11 compiling [doc](https://pybind11.readthedocs.io/en/stable/compiling.html).

The compiled libraries can be found in the `build` direcotory. To use the C++ libraries, 
one needs to specify the path of the library by:

```
export PYTHONPATH=$PYTHONPATH:<path-to-library>
```
