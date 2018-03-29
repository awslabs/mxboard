#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""Add or check license header

Usuage:

- add the default license header to source files that do not contain a valid
  license:

  python license_header.py add

- check if every files has a license header

  python license_header.py check
"""

import re
import os
import argparse
from itertools import chain
import logging
import sys

# the default apache license
_LICENSE = """Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

# if a file contains any str in the list, then consider it has been licensed
_LICENSE_PATTERNS = ['Licensed to the Apache Software Foundation']

# the folders or files that will be ignored
_WHITE_LIST = ['R-package/',
               'cub/',
               'docker/Dockerfiles',
               'dmlc-core/',
               'nnvm',
               '3rdparty',
               'src/operator/mkl/',
               'src/operator/special_functions-inl.h',
               'src/operator/nn/pool.h',
               'src/operator/contrib/psroi_pooling-inl.h',
               'src/operator/contrib/nn/deformable_im2col.h',
               'src/operator/contrib/nn/deformable_im2col.cuh',
               'src/operator/nn/im2col.h',
               'src/operator/nn/im2col.cuh',
               'example/ssd/dataset/pycocotools/coco.py',
               'example/rcnn/rcnn/cython/setup.py',
               'example/rcnn/rcnn/cython/nms_kernel.cu',
               'prepare_mkl.sh',
               'example/image-classification/predict-cpp/image-classification-predict.cc',
               'src/operator/contrib/ctc_include/']

# language extensions and the according commment mark
_LANGS = {'.cc':'*', '.h':'*', '.cu':'*', '.cuh':'*', '.py':'#',
          '.pm':'#', '.scala':'*', '.cc':'*', '.sh':'#', '.cmake':'#',
          '.java':'*', '.sh':'#', '.cpp':'*', '.hpp':'*', '.c':'*',
          '.bat':'rem', '.pl':'#', '.m':'%', '.R':'#', '.mk':'#', '.cfg':'#', '.t':'#'}

# Previous license header, which will be removed
_OLD_LICENSE = re.compile('.*Copyright.*by Contributors')

def _has_license(lines):
    return any([any([p in l for p in _LICENSE_PATTERNS]) for l in lines])

def _get_license(comment_mark):
    if comment_mark == '*':
        body = '/*\n'
    else:
        body = ''
    for l in _LICENSE.split('\n'):
        if comment_mark == '*':
            body += ' '
        body += comment_mark
        if len(l):
            body += ' ' + l
        body += '\n'

    if comment_mark == '*':
        body += ' */\n'
    body += '\n'
    return body

def _valid_file(fname, verbose=False):
    if any([l in fname for l in _WHITE_LIST]):
        if verbose:
            logging.info('skip ' + fname + ', it matches the white list')
        return False
    _, ext = os.path.splitext(fname)
    if ext not in _LANGS:
        if verbose:
            logging.info('skip ' + fname + ', unknown file extension')
        return False
    return True

def process_file(fname, action, verbose=True):
    if not _valid_file(fname, verbose):
        return True
    with open(fname, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return True
    if _has_license(lines):
        return True
    elif action == 'check':
        return False
    _, ext = os.path.splitext(fname)
    with open(fname, 'w', encoding="utf-8") as f:
        # shebang line
        if lines[0].startswith('#!'):
            f.write(lines[0].rstrip()+'\n\n')
            del lines[0]
        f.write(_get_license(_LANGS[ext]))
        for l in lines:
            f.write(l.rstrip()+'\n')
    logging.info('added license header to ' + fname)
    return True

def process_folder(root, action):
    excepts = []
    for root, _, files in os.walk(root):
        for f in files:
            fname = os.path.normpath(os.path.join(root, f))
            if not process_file(fname, action):
                excepts.append(fname)
    if action == 'check' and excepts:
        logging.warning('The following files do not contain a valid license, '+
                        'you can use `python tools/license_header.py add [file]` to add'+
                        'them automatically: ')
        for x in excepts:
            logging.warning(x)
        return False
    return True

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    parser = argparse.ArgumentParser(
        description='Add or check source license header')
    parser.add_argument(
        'action', nargs=1, type=str,
        choices=['add', 'check'], default='add',
        help = 'add or check')

    parser.add_argument(
        'file', nargs='*', type=str, action='append',
        help='Files to add license header to')

    args = parser.parse_args()
    files = list(chain(*args.file))
    action = args.action[0]
    has_license = True
    if len(files) > 0:
        for file in files:
            has_license = process_file(file, action)
            if action == 'check' and not has_license:
                logging.warn("{} doesn't have a license".format(file))
                has_license = False
    else:
        has_license = process_folder(os.path.join(os.path.dirname(__file__), '..'), action)
    if not has_license:
        sys.exit(1)
    else:
        sys.exit(0)

