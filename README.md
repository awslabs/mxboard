# MXBoard: Logging MXNet Data for Visualization in TensorBoard

## Overview

This project (MXBoard) provides Python APIs logging
[MXNet](http://mxnet.incubator.apache.org/) data for visualization in
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). 

### How to install TensorBoard
To launch TensorBoard for visualization, make sure you have the
[official release of TensorBoard](https://pypi.python.org/pypi/tensorboard) installed.
You can type `pip install tensorflow && pip install tensorboard`
on you machine to install TensorBoard.

### How to launch TensorBoard
After you installed the TensorBoard Python package, type the following command in the terminal
to launch TensorBoard:
```
tensorborad --logdir=/path/to/your/log/dir --host=your_host_ip --port=your_port_number
```
As an example of visualizing data using the browser on your machine, you can type
```
tensorborad --logdir=/path/to/your/log/dir --host=127.0.0.1 --port=8888
```
Then in the browser, type address `127.0.0.1:8888`. Note that in some situations,
the port number `8888` may be occupied by other applications and launching TensorBoard
may fail. You may choose a different port number that is available in those situations.


### How to use TensorBoard GUI for data visualization
Please find the tutorials on
[TensorFlow website](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
for details.

### What are other required packages for using MXBoard
Please make sure the following Python packages have been installed before using
the logging APIs:
- [MXNet](https://pypi.python.org/pypi/mxnet)
- [protobuf3](https://pypi.python.org/pypi/protobuf)
- [six](https://pypi.python.org/pypi/six)
- [Pillow](https://pypi.python.org/pypi/Pillow)


### What data types in TensorBoard GUI are supported by MXNet logging APIs
We currently support the following data types that you can find on the TensorBoard GUI:
- SCALARS
- IMAGES
- [HISTOGRAMS](https://www.tensorflow.org/programmers_guide/tensorboard_histograms)
- [PROJECTOR/EMBENDDING](https://www.tensorflow.org/programmers_guide/embedding)
- AUDIO
- TEXT
- PR CURVES


MXBoard provides the logging APIs through the `SummaryWriter` class.

```python
    mxboard.SummaryWriter
    mxboard.SummaryWriter.add_audio
    mxboard.SummaryWriter.add_embedding
    mxboard.SummaryWriter.add_histogram
    mxboard.SummaryWriter.add_image
    mxboard.SummaryWriter.add_pr_curve
    mxboard.SummaryWriter.add_scalar
    mxboard.SummaryWriter.add_text
    mxboard.SummaryWriter.close
    mxboard.SummaryWriter.flush
    mxboard.SummaryWriter.get_logdir
    mxboard.SummaryWriter.reopen
```

## Examples
Let's take a look at several simple examples demonstrating the use MXBoard logging APIs.

### Scalar
Scalar values are often plotted in terms of curves, such as training accuracy as time evolves. Here
is an example of plotting the curve of `y=sin(x/100)` where `x` is in the range of `[0, 2*pi]`.
```python
import numpy as np
from mxboard import SummaryWriter

x_vals = np.arange(start=0, stop=2 * np.pi, step=0.01)
y_vals = np.sin(x_vals)
with SummaryWriter(logdir='./logs') as sw:
    for x, y in zip(x_vals, y_vals):
        sw.add_scalar(tag='sin_function_curve', value=y, global_step=x * 100)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_scalar_sin.png)


### Histogram
We can visulize the value distributions of tensors by logging `NDArray`s in terms of histograms.
The following code snippet generates a series of normal distributions with smaller and smaller standard deviations.
```python
import mxnet as mx
from mxboard import SummaryWriter


with SummaryWriter(logdir='./logs') as sw:
    for i in range(10):
        # create a normal distribution with fixed mean and decreasing std
        data = mx.nd.normal(loc=0, scale=10.0/(i+1), shape=(10, 3, 8, 8))
        sw.add_histogram(tag='norml_dist', values=data, bins=200, global_step=i)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_histogram_norm.png)


### Image
The image logging API can take MXNet `NDArray` or `numpy.ndarray` of 2-4 dimensions.
It will preprocess the input image and write the processed image to the event file.
When the input image data is 2D or 3D, it represents a single image.
When the input image data is a 4D tensor, which represents a batch of images, the logging
API would make a grid of those images by stitching them together before write
them to the event file. The following code snippet saves 15 same images
for visualization in TensorBoard.
```python
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter
from scipy import misc

# get a racoon face image from scipy
# and convert its layout from HWC to CHW
face = misc.face().transpose((2, 0, 1))
# add a batch axis in the beginning
face = face.reshape((1,) + face.shape)
# replicate the face by 15 times
faces = [face] * 15
# concatenate the faces along the batch axis
faces = np.concatenate(faces, axis=0)

img = mx.nd.array(faces, dtype=faces.dtype)
with SummaryWriter(logdir='./logs') as sw:
    # write batched faces to the event file
    sw.add_image(tag='faces', image=img)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_image_faces.png)


### Embedding
Embedding visualization enables people to get an intuition on how data is clustered
in 2D or 3D space. The following code takes 2,560 images of handwritten digits
from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and log them
as embedding vectors with labels and original images.
```python
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxboard import SummaryWriter


batch_size = 128


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label

# training dataset containing MNIST images and labels
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

initialized = False
embedding = None
labels = None
images = None

for i, (data, label) in enumerate(train_data):
    if i >= 20:
        # only fetch the first 20 batches of images
        break
    if initialized:  # after the first batch, concatenate the current batch with the existing one
        embedding = mx.nd.concat(*(embedding, data), dim=0)
        labels = mx.nd.concat(*(labels, label), dim=0)
        images = mx.nd.concat(*(images, data.reshape(batch_size, 1, 28, 28)), dim=0)
    else:  # first batch of images, directly assign
        embedding = data
        labels = label
        images = data.reshape(batch_size, 1, 28, 28)
        initialized = True

with SummaryWriter(logdir='./logs') as sw:
    sw.add_embedding(tag='mnist', embedding=embedding, labels=labels, images=images)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_embedding_mnist.png)


### Audio
The following code generates audio data uniformly sampled in range `[-1, 1]`
and write the data to the event file for TensorBoard to playback.
```python
import mxnet as mx
from mxboard import SummaryWriter


frequency = 44100
# 44100 random samples between -1 and 1
data = mx.random.uniform(low=-1, high=1, shape=(frequency,))
max_abs_val = data.abs().max()
# rescale the data to the range [-1, 1]
data = data / max_abs_val
with SummaryWriter(logdir='./logs') as sw:
    sw.add_audio(tag='uniform_audio', audio=data, global_step=0)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_audio_uniform.png)


### Text
TensorBoard is able to render plain text as well as text in the markdown format.
The following code demonstrates these two use cases.
```python
from mxboard import SummaryWriter


def simple_example(sw, step):
    greeting = 'Hello MXNet from step {}'.format(str(step))
    sw.add_text(tag='simple_example', text=greeting, global_step=step)


def markdown_table(sw):
    header_row = 'Hello | MXNet,\n'
    delimiter = '----- | -----\n'
    table_body = 'This | is\n' + 'so | awesome!'
    sw.add_text(tag='markdown_table', text=header_row+delimiter+table_body)


with SummaryWriter(logdir='./logs') as sw:
    simple_example(sw, 100)
    markdown_table(sw)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_text.png)


### PR Curve
Precision-Recall is a useful metric of success of prediction when the categories are imbalanced.
The relationship between recall and precision can be visualized in terms of precision-recall curves.
The following code snippet logs the data of predictions and labels for visualizing
the precision-recall curve in TensorBoard. It generates 100 numbers uniformly distributed in range `[0, 1]` representing
the predictions of 100 examples. The labels are also generated randomly by picking either 0 or 1.
```python
import mxnet as mx
import numpy as np
from mxboard import SummaryWriter

with SummaryWriter(logdir='./logs') as sw:
    predictions = mx.nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
    labels = mx.nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
    sw.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_pr_curve_uniform.png)


## References
1. https://github.com/TeamHG-Memex/tensorboard_logger
2. https://github.com/lanpa/tensorboard-pytorch
3. https://github.com/dmlc/tensorboard
4. https://github.com/tensorflow/tensorflow
5. https://github.com/tensorflow/tensorboard

## License

This library is licensed under the Apache 2.0 License. 
