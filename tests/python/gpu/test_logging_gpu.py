from __future__ import print_function
import sys
import os
import mxnet as mx
from mxnet.test_utils import set_default_context

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed

from test_logging import *

set_default_context(mx.gpu(0))

if __name__ == '__main__':
    import nose
    nose.runmodule()
