## Training an MNIST model with MXBoard
In this example, we borrow the training script for an MNIST model
from [MXNet Gluon](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/mnist.py),
and apply the MXBoard logging APIs to visualize accuracies, images, and gradients while training
is in progress.
Search for variable `sw`, which is an object instance of `SummaryWriter`, in the script
to understand how it plays.

The `SummaryWriter` object instance is used as the following:

1. Create a `SummaryWriter` object instance for logging. This is the line
```python
sw = SummaryWriter(logdir='logs', flush_secs=5)
```
Here we specify the logging directory as `logs` under the current folder and flushing
data to event files every five seconds in order to see training progress more promptly.

2. Log cross entropy loss. This is the line
```python
sw.add_scalar(tag='cross_entropy', value=L.mean().asscalar(), global_step=global_step)
```

3. Log the first batch image for each epoch to verify that dataset is really shuffled
for each epoch. This is the line
```python
sw.add_image('minist_first_minibatch', data.reshape((opt.batch_size, 1, 28, 28)), epoch)
```

4. Collect the gradients of all the parameters and log them as histograms to verify
the training algorithm is converging. When training reaches the optimal solution, most of
the gradients of parameters will be centered near zero. This is the line
```python
sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)
```

5. Save training accuracy for each epoch and plot the accuracy curve. This is the line
```python
sw.add_scalar(tag='train_acc', value=acc, global_step=epoch)
```

6. Save validation accuracy for each epoch and plot the accuracy curve. This is the line
```python
sw.add_scalar(tag='valid_acc', value=val_acc, global_step=epoch)
```

7. Remember to close the `SummaryWriter` in the end.
```python
sw.close()
```