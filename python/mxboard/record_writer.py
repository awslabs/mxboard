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

"""Writer for writing events to the event file."""

from __future__ import absolute_import
import struct
from ._crc32c import crc32c


class RecordWriter(object):
    """Write records in the following format for a single record event_str:
    uint64 len(event_str)
    uint32 masked crc of len(event_str)
    byte event_str
    uint32 masked crc of event_str
    The implementation is ported from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_writer.cc
    Here we simply define a byte string _dest to buffer the record to be written to files.
    The flush and close mechanism is totally controlled in this class.
    In TensorFlow, _dest is a object instance of ZlibOutputBuffer (C++) which has its own flush
    and close mechanism defined."""
    def __init__(self, path):
        self._writer = None
        try:
            self._writer = open(path, 'wb')
        except (OSError, IOError) as err:
            raise ValueError('failed to open file {}: {}'.format(path, str(err)))

    def __del__(self):
        self.close()

    def write_record(self, event_str):
        """Writes a serialized event to file."""
        header = struct.pack('Q', len(event_str))
        header += struct.pack('I', masked_crc32c(header))
        footer = struct.pack('I', masked_crc32c(event_str))
        self._writer.write(header + event_str + footer)

    def flush(self):
        """Flushes the event string to file."""
        assert self._writer is not None
        self._writer.flush()

    def close(self):
        """Closes the record writer."""
        if self._writer is not None:
            self.flush()
            self._writer.close()
            self._writer = None


def masked_crc32c(data):
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    x = u32(crc32c(data))  # pylint: disable=invalid-name
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):  # pylint: disable=invalid-name
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    return x & 0xffffffff
