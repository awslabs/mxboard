# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""

from __future__ import absolute_import

import os
import sys
from setuptools import find_packages

if "--inplace" in sys.argv:
    from distutils.core import setup
    kwargs = {}
else:
    from setuptools import setup
    kwargs = {'install_requires': ['numpy', 'protobuf>=3.0.0', 'Pillow', 'six'],
              'zip_safe': False}


def compile_summary_protobuf():
    proto_path = 'mxboard/proto'
    proto_files = os.path.join(proto_path, '*.proto')
    cmd = 'protoc ' + proto_files + ' --python_out=.'
    print('compiling protobuf files in {}'.format(proto_path))
    return os.system('set -ex &&' + cmd)


if compile_summary_protobuf() != 0:
    print('ERROR: Compiling summary protocol buffers failed. You will not be '
          'able to use the logging APIs for visualizing MXNet data in TensorBoard. '
          'Please make sure that you have installed protobuf3 compiler and runtime correctly.')
    sys.exit(1)


setup(
    name='mxboard',
    version='0.1.1',
    description='A logging tool for visualizing MXNet data in TensorBoard',
    long_description='MXBoard is a logging tool that enables visualization of MXNet data in TensorBoard.',
    author='Amazon Web Services',
    author_email='jwum@amazon.com',
    url='https://github.com/awslabs/mxboard',
    packages=find_packages(),
    include_package_data=True,
    license='Apache License 2.0',
    test_suite='tests',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='mxnet tensor visualization machine deep learning',
    **kwargs)
