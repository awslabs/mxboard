#!/bin/bash
pylint --rcfile=tests/ci_build/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$,.*\.proto$$" python/mxboard/*.py python/mxboard/proto/__init__.py
