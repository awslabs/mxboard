MXBoard Python Package
====================
This directory and nested files contain MXBoard Python package.

## Installation
To install MXBoard Python package, visit MXBoard
[Install Instruction](https://github.com/awslabs/mxboard/tree/tensorboard_logging#installation)


## Running the unit tests

For running unit tests, you will need the [nose PyPi package](https://pypi.python.org/pypi/nose).
To install:
```bash
pip install --upgrade nose
```

Once ```nose``` is installed, run the following from MXBoard root directory:
```
nosetests --verbose tests/python/unittest
nosetests --verbose tests/python/gpu
```