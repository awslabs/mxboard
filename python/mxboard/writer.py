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

"""APIs for logging data in the event file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import os
import logging
from .proto import event_pb2
from .proto import summary_pb2
from .event_file_writer import EventFileWriter
from .summary import scalar_summary, histogram_summary, image_summary, audio_summary
from .summary import text_summary, pr_curve_summary, _net2pb
from .utils import _save_embedding_tsv, _make_sprite_image, _make_metadata_tsv
from .utils import _add_embedding_config, _make_numpy_array, _get_embedding_dir


class SummaryToEventTransformer(object):
    """This class is adapted with minor modifications from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py#L125
    Users should not use this class directly for logging MXNet data.
    This class abstractly implements the SummaryWriter API: add_summary.
    The endpoint generates an event protobuf from the Summary object, and passes
    the event protobuf to _event_writer, which is of type EventFileWriter, for logging.
    """
    # TODO(junwu): Need to check its compatibility with using ONNX for visualizing MXNet graphs.
    def __init__(self, event_writer):
        """Initializes the _event_writer with the passed-in value.

        Parameters
        ----------
          event_writer: EventFileWriter
              An event file writer writing events to the files in the path `logdir`.
        """
        self._event_writer = event_writer
        # This set contains tags of Summary Values that have been encountered
        # already. The motivation here is that the SummaryWriter only keeps the
        # metadata property (which is a SummaryMetadata proto) of the first Summary
        # Value encountered for each tag. The SummaryWriter strips away the
        # SummaryMetadata for all subsequent Summary Values with tags seen
        # previously. This saves space.
        self._seen_summary_tags = set()

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer and adds it
        to the event file.

        Parameters
        ----------
          summary : A `Summary` protocol buffer
              Optionally serialized as a string.
          global_step: Number
              Optional global step value to record with the summary.
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        # We strip metadata from values with tags that we have seen before in order
        # to save space - we just store the metadata on the first value with a
        # specific tag.
        for value in summary.value:
            if not value.metadata:
                continue

            if value.tag in self._seen_summary_tags:
                # This tag has been encountered before. Strip the metadata.
                value.ClearField("metadata")
                continue

            # We encounter a value with a tag we have not encountered previously. And
            # it has metadata. Remember to strip metadata from future values with this
            # tag string.
            self._seen_summary_tags.add(value.tag)

        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    def add_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file."""
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
    """This class is adapted from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py.
    Even though this class provides user-level APIs in TensorFlow, it is recommended to use the
    interfaces defined in the class `SummaryWriter` (see below) for logging in MXNet as they are
    directly compatible with the MXNet NDArray type.
    This class writes `Summary` protocol buffers to event files. The `FileWriter` class provides
    a mechanism to create an event file in a given directory and add summaries and events to it.
    The class updates the file contents asynchronously.
    """
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None, verbose=True):
        """Creates a `FileWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, or `add_event()`.

        Parameters
        ----------
            logdir : str
                Directory where event file will be written.
            max_queue : int
                Size of the queue for pending events and summaries.
            flush_secs: Number
                How often, in seconds, to flush the pending events and summaries to disk.
            filename_suffix : str
                Every event file's name is suffixed with `filename_suffix` if provided.
            verbose : bool
                Determines whether to print logging messages.
        """
        event_writer = EventFileWriter(logdir, max_queue, flush_secs, filename_suffix, verbose)
        super(FileWriter, self).__init__(event_writer)

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.

        Parameters
        ----------
            event : An `Event` protocol buffer.
        """
        self._event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file. Does nothing if the EventFileWriter
        was not closed.
        """
        self._event_writer.reopen()


class SummaryWriter(object):
    """This class is adapted with modifications in support of the MXNet NDArray types from
    https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py.
    The `SummaryWriter` class provides a high-level api to create an event file in a
    given directory and add summaries and events to it. This class writes data to the
    event file asynchronously.
    This class is a wrapper of the FileWriter class. It's recommended that users use
    the APIs of this class to log MXNet data for visualization as they are directly compatible with
    the MXNet data types.

    Examples
    --------
    >>> data = mx.nd.random.uniform(size=(10, 10))
    >>> with SummaryWriter(logdir='logs') as sw:
    >>>     sw.add_histogram(tag='my_hist', values=data, global_step=0, bins=100)
    """
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None, verbose=True):
        """
        Creates a `SummaryWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_audio()`, `add_embedding()`,
        `add_histogram()`, `add_image()`, `add_pr_curve()`, `add_scalar()`, and `add_text()`.
        Please make sure that the `logdir` used here for initiailizing `SummaryWriter`
        matches the `--logdir` parameter you passed to the `tensorboard` binary in the command line
        for launching TensorBoard.

        Parameters
        ----------
            logdir : str
                Directory where event file will be written.
            max_queue : int
                Size of the queue for pending events and summaries.
            flush_secs: Number
                How often, in seconds, to flush the pending events and summaries to disk.
            filename_suffix : str
                Every event file's name is suffixed with `filename_suffix` if provided.
            verbose : bool
                Determines whether to print the logging messages.
        """
        self._file_writer = FileWriter(logdir=logdir, max_queue=max_queue,
                                       flush_secs=flush_secs, filename_suffix=filename_suffix,
                                       verbose=verbose)
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._filename_suffix = filename_suffix
        self._verbose = verbose
        # for writing scalars of different tags in the same plot
        self._all_writers = {self._file_writer.get_logdir(): self._file_writer}
        self._logger = None
        if verbose:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
        self._default_bins = None
        self._text_tags = []
        # scalar value dict.
        # key: file_writer's logdir, value: list of [timestamp, global_step, value]
        self._scalar_dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_default_bins(self):
        """Ported from the C++ function InitDefaultBucketsInner() in the following file.
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/histogram/histogram.cc
        See the following tutorial for more details on how TensorFlow initialize bin distribution.
        https://www.tensorflow.org/programmers_guide/tensorboard_histograms"""
        if self._default_bins is None:
            v = 1E-12
            buckets = []
            neg_buckets = []
            while v < 1E20:
                buckets.append(v)
                neg_buckets.append(-v)
                v *= 1.1
            self._default_bins = neg_buckets[::-1] + [0] + buckets
        return self._default_bins

    def _append_to_scalar_dict(self, tag, scalar_value, global_step, timestamp):
        """Adds a list [timestamp, step, value] to the value of `self._scalar_dict[tag]`.
        This allows users to store scalars in memory and dump them to a json file later."""
        if tag not in self._scalar_dict.keys():
            self._scalar_dict[tag] = []
        self._scalar_dict[tag].append([timestamp, global_step, float(scalar_value)])

    def get_logdir(self):
        """Returns the logging directory associated with this `SummaryWriter`."""
        return self._file_writer.get_logdir()

    def add_scalar(self, tag, value, global_step=None):
        """Adds scalar data to the event file.

        Parameters
        ----------
            tag : str
                Name for the scalar plot.
            value : float, tuple, list, or dict
                If value is a float, the corresponding curve would have no name attached in the
                plot.
                If value is a tuple or list, it must have two elements with the first one
                representing the name of the value and the second one as the float value. The
                name of the value will be attached to the corresponding curve in the plot. This
                is useful when users want to draw multiple curves in the same plot. It internally
                calls `_add_scalars`.
                If value is a dict, it's a mapping from strs to float values, with strs
                representing the names of the float values. This is convenient when users want
                to log a collection of float values with different names for visualizing them in
                the same plot without repeatedly calling `add_scalar` for each value. It internally
                calls `_add_scalars`.
            global_step : int
                Global step value to record.

        Examples
        --------
        >>> import numpy as np
        >>> from mxboard import SummaryWriter

        >>> xs = np.arange(start=0, stop=2 * np.pi, step=0.01)
        >>> y_sin = np.sin(xs)
        >>> y_cos = np.cos(xs)
        >>> y_exp_sin = np.exp(y_sin)
        >>> y_exp_cos = np.exp(y_cos)
        >>> y_sin2 = y_sin * y_sin
        >>> with SummaryWriter(logdir='./logs') as sw:
        >>>     for x, y1, y2, y3, y4, y5 in zip(xs, y_sin, y_cos, y_exp_sin, y_exp_cos, y_sin2):
        >>>         sw.add_scalar('curves', {'sin': y1, 'cos': y2}, x * 100)
        >>>         sw.add_scalar('curves', ('exp(sin)', y3), x * 100)
        >>>         sw.add_scalar('curves', ['exp(cos)', y4], x * 100)
        >>>         sw.add_scalar('curves', y5, x * 100)
        """
        if isinstance(value, (tuple, list, dict)):
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError('expected two elements in value, while received %d'
                                     % len(value))
                value = {value[0]: value[1]}
            self._add_scalars(tag, value, global_step)
        else:
            self._file_writer.add_summary(scalar_summary(tag, value), global_step)
            self._append_to_scalar_dict(self.get_logdir() + '/' + tag,
                                        value, global_step, time.time())

    def _add_scalars(self, tag, scalar_dict, global_step=None):
        """Adds multiple scalars to summary. This enables drawing multiple curves in one plot.

        Parameters
        ----------
            tag : str
                Name for the plot.
            scalar_dict : dict
                Values to be saved.
            global_step : int
                Global step value to record.
        """
        timestamp = time.time()
        fw_logdir = self._file_writer.get_logdir()
        for scalar_name, scalar_value in scalar_dict.items():
            fw_tag = fw_logdir + '/' + tag + '/' + scalar_name
            if fw_tag in self._all_writers.keys():
                fw = self._all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag, max_queue=self._max_queue,
                                flush_secs=self._flush_secs, filename_suffix=self._filename_suffix,
                                verbose=self._verbose)
                self._all_writers[fw_tag] = fw
            fw.add_summary(scalar_summary(tag, scalar_value), global_step)
            self._append_to_scalar_dict(fw_tag, scalar_value, global_step, timestamp)

    def export_scalars(self, path):
        """Exports to the given path an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}
        """
        if os.path.exists(path) and os.path.isfile(path):
            logging.warning('%s already exists and will be overwritten by scalar dict', path)
        with open(path, "w") as f:
            json.dump(self._scalar_dict, f)

    def clear_scalar_dict(self):
        """Empties scalar dictionary."""
        self._scalar_dict = {}

    def add_histogram(self, tag, values, global_step=None, bins='default'):
        """Add histogram data to the event file.

        Note: This function internally calls `asnumpy()` if `values` is an MXNet NDArray.
        Since `asnumpy()` is a blocking function call, this function would block the main
        thread till it returns. It may consequently affect the performance of async execution
        of the MXNet engine.

        Parameters
        ----------
            tag : str
                Name for the `values`.
            values : MXNet `NDArray` or `numpy.ndarray`
                Values for building histogram.
            global_step : int
                Global step value to record.
            bins : int or sequence of scalars or str
                If `bins` is an int, it defines the number equal-width bins in the range
                `(values.min(), values.max())`.
                If `bins` is a sequence, it defines the bin edges, including the rightmost edge,
                allowing for non-uniform bin width.
                If `bins` is a str equal to 'default', it will use the bin distribution
                defined in TensorFlow for building histogram.
                Ref: https://www.tensorflow.org/programmers_guide/tensorboard_histograms
                The rest of supported strings for `bins` are 'auto', 'fd', 'doane', 'scott',
                'rice', 'sturges', and 'sqrt'. etc. See the documentation of `numpy.histogram`
                for detailed definitions of those strings.
                https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        if bins == 'default':
            bins = self._get_default_bins()
        self._file_writer.add_summary(histogram_summary(tag, values, bins), global_step)

    def add_image(self, tag, image, global_step=None):
        """Add image data to the event file.
        This function supports input as a 2D, 3D, or 4D image.
        If the input image is 2D, a channel axis is prepended as the first dimension
        and image will be replicated three times and concatenated along the channel axis.
        If the input image is 3D, it will be replicated three times and concatenated along
        the channel axis. If the input image is 4D, which is a batch images, all the
        images will be spliced as a sprite image for display.

        Note: This function requires the ``pillow`` package.

        Note: This function internally calls `asnumpy()` for MXNet `NDArray` inputs.
        Since `asnumpy()` is a blocking function call, this function would block the main
        thread till it returns. It may consequently affect the performance of async execution
        of the MXNet engine.

        Parameters
        ----------
            tag : str
                Name for the `image`.
            image : MXNet `NDArray` or `numpy.ndarray`
                Image is one of the following formats: (H, W), (C, H, W), (N, C, H, W).
                If the input is a batch of images, a grid of images is made by stitching them
                together.
                If data type is float, values must be in range [0, 1], and then they are
                rescaled to range [0, 255]. Note that this does not change the values of the
                input `image`. A copy of the input `image` is created instead.
                If data type is 'uint8`, values are unchanged.
            global_step : int
                Global step value to record.
        """
        self._file_writer.add_summary(image_summary(tag, image), global_step)

    def add_audio(self, tag, audio, sample_rate=44100, global_step=None):
        """Add audio data to the event file.

        Note: This function internally calls `asnumpy()` for MXNet `NDArray` inputs.
        Since `asnumpy()` is a blocking function call, this function would block the main
        thread till it returns. It may consequently affect the performance of async execution
        of the MXNet engine.

        Parameters
        ----------
            tag : str
                Name for the `audio`.
            audio : MXNet `NDArray` or `numpy.ndarray`
                Audio data squeezable to a 1D tensor. The values of the tensor are in the range
                `[-1, 1]`.
            sample_rate : int
                Sample rate in Hz.
            global_step : int
                Global step value to record.
        """
        self._file_writer.add_summary(audio_summary(tag, audio, sample_rate=sample_rate),
                                      global_step)

    def add_text(self, tag, text, global_step=None):
        """Add text data to the event file.

        Parameters
        ----------
            tag : str
                Name for the `text`.
            text : str
                Text to be saved to the event file.
            global_step : int
                Global step value to record.
        """
        self._file_writer.add_summary(text_summary(tag, text), global_step)
        if tag not in self._text_tags:
            self._text_tags.append(tag)
            extension_dir = self.get_logdir() + '/plugins/tensorboard_text/'
            if not os.path.exists(extension_dir):
                os.makedirs(extension_dir)
            with open(extension_dir + 'tensors.json', 'w') as fp:
                json.dump(self._text_tags, fp)

    def add_embedding(self, tag, embedding, labels=None, images=None, global_step=None):
        """Adds embedding projector data to the event file. It will also create a config file
        used by the embedding projector in TensorBoard. The folder containing the embedding
        data is named using the formula:
        If global_step is not None, the folder name is `tag + '_' + str(global_step).zfill(6)`;
        else, the folder name is `tag`.
        For example, tag = 'mnist', global_step = 12, the folder's name is 'mnist_000012';
        when global_step = None, the folder's name is 'mnist'.
        See the following reference for the meanings of labels and images.
        Ref: https://www.tensorflow.org/versions/r1.2/get_started/embedding_viz

        Note: This function internally calls `asnumpy()` for MXNet `NDArray` inputs.
        Since `asnumpy()` is a blocking function call, this function would block the main
        thread till it returns. It may consequently affect the performance of async execution
        of the MXNet engine.

        Parameters
        ----------
            tag : str
                Name for the `embedding`.
            embedding : MXNet `NDArray` or  `numpy.ndarray`
                A matrix whose each row is the feature vector of a data point.
            labels : MXNet `NDArray` or `numpy.ndarray` or a list of elements convertible to str.
                Labels corresponding to the data points in the `embedding`.
            images : MXNet `NDArray` or `numpy.ndarray`
                Images of format NCHW corresponding to the data points in the `embedding`.
            global_step : int
                Global step value to record. If not set, default to zero.
        """
        embedding_shape = embedding.shape
        if len(embedding_shape) != 2:
            raise ValueError('expected 2D NDArray as embedding data, while received an array with'
                             ' ndim=%d' % len(embedding_shape))
        data_dir = _get_embedding_dir(tag, global_step)
        save_path = os.path.join(self.get_logdir(), data_dir)
        try:
            os.makedirs(save_path)
        except OSError:
            logging.warning('embedding dir %s exists, files under this dir will be overwritten',
                            save_path)
        if labels is not None:
            if embedding_shape[0] != len(labels):
                raise ValueError('expected equal values of embedding first dim and length of '
                                 'labels, while received %d and %d for each'
                                 % (embedding_shape[0], len(labels)))
            if self._logger is not None:
                self._logger.info('saved embedding labels to %s', save_path)
            _make_metadata_tsv(labels, save_path)
        if images is not None:
            img_labels_shape = images.shape
            if embedding_shape[0] != img_labels_shape[0]:
                raise ValueError('expected equal first dim size of embedding and images,'
                                 ' while received %d and %d for each' % (embedding_shape[0],
                                                                         img_labels_shape[0]))
            if self._logger is not None:
                self._logger.info('saved embedding images to %s', save_path)
            _make_sprite_image(images, save_path)
        if self._logger is not None:
            self._logger.info('saved embedding data to %s', save_path)
        _save_embedding_tsv(embedding, save_path)
        _add_embedding_config(self.get_logdir(), data_dir, labels is not None,
                              images.shape if images is not None else None)

    def add_pr_curve(self, tag, labels, predictions, num_thresholds,
                     global_step=None, weights=None):
        """Adds precision-recall curve.

        Note: This function internally calls `asnumpy()` for MXNet `NDArray` inputs.
        Since `asnumpy()` is a blocking function call, this function would block the main
        thread till it returns. It may consequently affect the performance of async execution
        of the MXNet engine.

        Parameters
        ----------
            tag : str
                A tag attached to the summary. Used by TensorBoard for organization.
            labels : MXNet `NDArray` or `numpy.ndarray`.
                The ground truth values. A tensor of 0/1 values with arbitrary shape.
            predictions : MXNet `NDArray` or `numpy.ndarray`.
                A float32 tensor whose values are in the range `[0, 1]`. Dimensions must match
                those of `labels`.
            num_thresholds : int
                Number of thresholds, evenly distributed in `[0, 1]`, to compute PR metrics for.
                Should be `>= 2`. This value should be a constant integer value, not a tensor
                that stores an integer.
                The thresholds for computing the pr curves are calculated in the following way:
                `width = 1.0 / (num_thresholds - 1),
                thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]`.
            global_step : int
                Global step value to record.
            weights : MXNet `NDArray` or `numpy.ndarray`.
                Optional float32 tensor. Individual counts are multiplied by this value.
                This tensor must be either the same shape as or broadcastable to the `labels`
                tensor.
        """
        if num_thresholds < 2:
            raise ValueError('num_thresholds must be >= 2')
        labels = _make_numpy_array(labels)
        predictions = _make_numpy_array(predictions)
        self._file_writer.add_summary(pr_curve_summary(tag, labels, predictions,
                                                       num_thresholds, weights), global_step)

    def add_graph(self, net):
        """Given a symbol representing the network structure of an MXNet model, write it the event
        file for visualization in TensorBoard. The parameters will be assigned with the scope
        name with the node name of the operator they belong to.

        Parameters
        ----------
            net : Symbol or HybridBlock
                An mxnet.symbol.Symbol or mxnet.gluon.HybridBlock object defining
                the structure of a network. If it's a HybridBlock, users must call hybridize()
                and forward() once before passing to this function.
        """
        self._file_writer.add_graph(_net2pb(net=net))

    def flush(self):
        """Flushes pending events to the file."""
        for fw in self._all_writers.values():
            fw.flush()

    def close(self):
        """Closes the event file for writing."""
        for fw in self._all_writers.values():
            fw.close()

    def reopen(self):
        """Reopens the event file for writing."""
        for fw in self._all_writers.values():
            fw.reopen()
