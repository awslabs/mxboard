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

"""Functions of generating summary protocol buffers. Adapted from
https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/summary.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import io
import wave
import struct
import json
import re as _re
import numpy as np

try:
    import mxnet
    from distutils.version import LooseVersion
    if LooseVersion(mxnet.__version__) < LooseVersion('1.2.0'):
        logging.warning('The currently installed MXNet version %s is less than 1.2.0.'
                        ' Some functionality of MXBoard may not work.', mxnet.__version__)
except ImportError:
    raise ImportError('MXBoard requires MXNet with version >= 1.2.0.'
                      ' Please follow the instruction here to install MXNet first.'
                      ' http://mxnet.incubator.apache.org/install/index.html')

from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol
from mxnet.gluon import HybridBlock
from .proto.summary_pb2 import Summary
from .proto.summary_pb2 import HistogramProto
from .proto.summary_pb2 import SummaryMetadata
from .proto.tensor_pb2 import TensorProto
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.plugin_pr_curve_pb2 import PrCurvePluginData
from .proto.node_def_pb2 import NodeDef
from .proto.graph_pb2 import GraphDef
from .proto.attr_value_pb2 import AttrValue
from .proto.versions_pb2 import VersionDef
from .utils import _make_numpy_array, _prepare_image
try:
    from PIL import Image
except ImportError:
    Image = None


_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')


def _clean_tag(name):
    """Cleans a tag. Removes illegal characters for instance.
    Adapted from the TensorFlow function `clean_tag()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/summary_op_util.py

    Parameters
    ----------
        name : str
            The original tag name to be processed.

    Returns
    -------
        The cleaned tag name.
    """
    # In the past, the first argument to summary ops was a tag, which allowed
    # arbitrary characters. Now we are changing the first argument to be the node
    # name. This has a number of advantages (users of summary ops now can
    # take advantage of the tf name scope system) but risks breaking existing
    # usage, because a much smaller set of characters are allowed in node names.
    # This function replaces all illegal characters with _s, and logs a warning.
    # It also strips leading slashes from the name.
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            logging.warning('Summary name %s is illegal; using %s instead.', name, new_name)
            name = new_name
    return name


def scalar_summary(tag, scalar):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Adapted from the TensorFlow function `scalar()` at
    https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/summary/summary.py

    Parameters
    ----------
      tag : str
          A name for the generated summary. Will also serve as the series name in TensorBoard.
      scalar : int, MXNet `NDArray`, or `numpy.ndarray`
          A scalar value or an ndarray of shape (1,).

    Returns
    -------
      A `Summary` protobuf of the `scalar` value.

    Raises
    ------
      ValueError: If the scalar has the wrong shape or type.
    """
    tag = _clean_tag(tag)
    scalar = _make_numpy_array(scalar)
    assert(scalar.squeeze().ndim == 0), 'scalar should be 0D'
    scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=tag, simple_value=scalar)])


def histogram_summary(tag, values, bins):
    """Outputs a `Summary` protocol buffer with a histogram.
    Adding a histogram summary makes it possible to visualize the data's distribution in
    TensorBoard. See detailed explanation of the TensorBoard histogram dashboard at
    https://www.tensorflow.org/get_started/tensorboard_histograms
    This op reports an `InvalidArgument` error if any value is not finite.
    Adapted from the TensorFlow function `histogram()` at
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/summary.py

    Parameters
    ----------
        tag : str
            A name for the summary of the histogram. Will also serve as a series name in
            TensorBoard.
        values : MXNet `NDArray` or `numpy.ndarray`
            Values for building the histogram.

    Returns
    -------
        A `Summary` protobuf of the histogram.
    """
    tag = _clean_tag(tag)
    values = _make_numpy_array(values)
    hist = _make_histogram(values.astype(float), bins)
    return Summary(value=[Summary.Value(tag=tag, histo=hist)])


def _make_histogram(values, bins):
    """Converts values into a histogram proto using logic from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/histogram/histogram.cc"""
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    limits = limits[1:]

    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits,
                          bucket=counts)


def image_summary(tag, image):
    """Outputs a `Summary` protocol buffer with image(s).

    Parameters
    ----------
        tag : str
            A name for the generated summary. Will also serve as a series name in TensorBoard.
        image : MXNet `NDArray` or `numpy.ndarray`
            Image data that is one of the following layout: (H, W), (C, H, W), (N, C, H, W).
            The pixel values of the image are assumed to be normalized in the range [0, 1].
            The image will be rescaled to the range [0, 255] and cast to `np.uint8` before creating
            the image protobuf.

    Returns
    -------
        A `Summary` protobuf of the image.
    """
    tag = _clean_tag(tag)
    image = _prepare_image(image)
    image = _make_image(image)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def _make_image(tensor):
    """Converts an NDArray type image to Image protobuf"""
    assert isinstance(tensor, NDArray)
    if Image is None:
        raise ImportError('need to install PIL for visualizing images')
    height, width, channel = tensor.shape
    tensor = _make_numpy_array(tensor)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height, width=width, colorspace=channel,
                         encoded_image_string=image_string)


def audio_summary(tag, audio, sample_rate=44100):
    """Outputs a `Summary` protocol buffer with audio data.

    Parameters
    ----------
        tag : str
            A name for the generated summary. Will also serve as a series name in TensorBoard.
        audio : MXNet `NDArray` or `numpy.ndarray`
            Audio data that can be squeezed into 1D array. The values are in the range [-1, 1].
        sample_rate : int
            Sampling frequency. 44,100Hz is a common sampling frequency.

    Returns
    -------
        A `Summary` protobuf of the audio data.
    """
    audio = audio.squeeze()
    if audio.ndim != 1:
        raise ValueError('input audio must be squeezable to 1D, input audio squeezed '
                         'shape is {}'.format(audio.shape))
    audio = _make_numpy_array(audio)
    tensor_list = [int(32767.0 * x) for x in audio]
    fio = io.BytesIO()
    wave_writer = wave.open(fio, 'wb')
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(sample_rate)
    tensor_enc = b''
    for v in tensor_list:  # pylint: disable=invalid-name
        tensor_enc += struct.pack('<h', v)
    wave_writer.writeframes(tensor_enc)
    wave_writer.close()
    audio_string = fio.getvalue()
    fio.close()
    audio = Summary.Audio(sample_rate=sample_rate,
                          num_channels=1,
                          length_frames=len(tensor_list),
                          encoded_audio_string=audio_string,
                          content_type='audio/wav')
    return Summary(value=[Summary.Value(tag=tag, audio=audio)])


def text_summary(tag, text):
    """Outputs a `Summary` protocol buffer with audio data.

    Parameters
    ----------
        tag : str
            A name for the generated summary. Will also serve as a series name in TensorBoard.
        text : str
            Text data.

    Returns
    -------
        A `Summary` protobuf of the audio data.
    """
    plugin_data = [SummaryMetadata.PluginData(plugin_name='text')]
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[text.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(node_name=tag, metadata=smd, tensor=tensor)])


def pr_curve_summary(tag, labels, predictions, num_thresholds, weights=None):
    """Outputs a precision-recall curve `Summary` protocol buffer.

    Parameters
    ----------
        tag : str
            A tag attached to the summary. Used by TensorBoard for organization.
        labels : MXNet `NDArray` or `numpy.ndarray`.
            The ground truth values. A tensor of 0/1 values with arbitrary shape.
        predictions : MXNet `NDArray` or `numpy.ndarray`.
            A float32 tensor whose values are in the range `[0, 1]`. Dimensions must
            match those of `labels`.
        num_thresholds : int
            Number of thresholds, evenly distributed in `[0, 1]`, to compute PR metrics for.
            Should be `>= 2`. This value should be a constant integer value, not a tensor
            that stores an integer.
            The thresholds for computing the pr curves are calculated in the following way:
            `width = 1.0 / (num_thresholds - 1),
            thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]`.
        weights : MXNet `NDArray` or `numpy.ndarray`.
            Optional float32 tensor. Individual counts are multiplied by this value.
            This tensor must be either the same shape as or broadcastable to the `labels` tensor.

    Returns
    -------
        A `Summary` protobuf of the pr_curve.
    """
    # num_thresholds > 127 results in failure of creating protobuf,
    # probably a bug of protobuf
    if num_thresholds > 127:
        logging.warning('num_thresholds>127 would result in failure of creating pr_curve protobuf,'
                        ' clipping it at 127')
        num_thresholds = 127
    labels = _make_numpy_array(labels)
    predictions = _make_numpy_array(predictions)
    if weights is not None:
        weights = _make_numpy_array(weights)
    data = _compute_curve(labels, predictions, num_thresholds=num_thresholds, weights=weights)
    pr_curve_plugin_data = PrCurvePluginData(version=0,
                                             num_thresholds=num_thresholds).SerializeToString()
    plugin_data = [SummaryMetadata.PluginData(plugin_name='pr_curves',
                                              content=pr_curve_plugin_data)]
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(dtype='DT_FLOAT',
                         float_val=data.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=data.shape[0]),
                                  TensorShapeProto.Dim(size=data.shape[1])]))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])


# A value that we use as the minimum value during division of counts to prevent
# division by 0. 1.0 does not work: Certain weights could cause counts below 1.
_MINIMUM_COUNT = 1e-7


def _compute_curve(labels, predictions, num_thresholds, weights=None):
    """This function is another implementation of functions in
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py"""

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))


def _scoped_name(scope_name, node_name):
    return '/'.join([scope_name, node_name])


def _get_nodes_from_symbol(sym):
    """Given a symbol and shapes, return a list of `NodeDef`s for visualizing the
    the graph in TensorBoard."""
    if not isinstance(sym, Symbol):
        raise TypeError('sym must be an `mxnet.symbol.Symbol`,'
                        ' received type {}'.format(str(type(sym))))
    conf = json.loads(sym.tojson())
    nodes = conf['nodes']
    data2op = {}  # key: data id, value: list of ops to whom data is an input
    for i, node in enumerate(nodes):
        if node['op'] != 'null':  # node is an operator
            input_list = node['inputs']
            for idx in input_list:
                if idx[0] == 0:  # do not include 'data' node in the op scope
                    continue
                if idx[0] in data2op:
                    # nodes[idx[0]] is a data as an input to op nodes[i]
                    data2op[idx[0]].append(i)
                else:
                    data2op[idx[0]] = [i]

    # In the following, we group data with operators they belong to
    # by attaching them with operator names as scope names.
    # The parameters with the operator name as the prefix will be
    # assigned with the scope name of that operator. For example,
    # a convolution op has name 'conv', while its weight and bias
    # have name 'conv_weight' and 'conv_bias'. In the end, the operator
    # has scope name 'conv' prepended to its name, i.e. 'conv/conv'.
    # The parameters are named 'conv/conv_weight' and 'conv/conv_bias'.
    node_defs = []
    for i, node in enumerate(nodes):
        node_name = node['name']
        op_name = node['op']
        kwargs = {'op': op_name, 'name': node_name}
        if op_name != 'null':  # node is an operator
            inputs = []
            input_list = node['inputs']
            for idx in input_list:
                input_node = nodes[idx[0]]
                input_node_name = input_node['name']
                if input_node['op'] != 'null':
                    inputs.append(_scoped_name(input_node_name, input_node_name))
                elif idx[0] in data2op and len(data2op[idx[0]]) == 1 and data2op[idx[0]][0] == i:
                    # the data is only as an input to nodes[i], no else
                    inputs.append(_scoped_name(node_name, input_node_name))
                else:  # the data node has no scope name, e.g. 'data' as the input node
                    inputs.append(input_node_name)
            kwargs['input'] = inputs
            kwargs['name'] = _scoped_name(node_name, node_name)
        elif i in data2op and len(data2op[i]) == 1:
            # node is a data node belonging to one op, find out which operator this node belongs to
            op_node_name = nodes[data2op[i][0]]['name']
            kwargs['name'] = _scoped_name(op_node_name, node_name)

        if 'attrs' in node:
            # TensorBoard would escape quotation marks, replace it with space
            attr = json.dumps(node['attrs'], sort_keys=True).replace("\"", ' ')
            attr = {'param': AttrValue(s=attr.encode(encoding='utf-8'))}
            kwargs['attr'] = attr
        node_def = NodeDef(**kwargs)
        node_defs.append(node_def)
    return node_defs


def _sym2pb(sym):
    """Converts an MXNet symbol to its graph protobuf definition."""
    return GraphDef(node=_get_nodes_from_symbol(sym), versions=VersionDef(producer=100))


def _net2pb(net):
    if isinstance(net, HybridBlock):
        # TODO(junwu): may need a more approprite way to get symbol from a HybridBlock
        if not net._cached_graph:
            raise RuntimeError(
                "Please first call net.hybridize() and then run forward with "
                "this net at least once before calling add_graph().")
        net = net._cached_graph[1]
    elif not isinstance(net, Symbol):
        raise TypeError('only accepts mxnet.gluon.HybridBlock and mxnet.symbol.Symbol '
                        'as input network, received type {}'.format(str(type(net))))
    return _sym2pb(net)
