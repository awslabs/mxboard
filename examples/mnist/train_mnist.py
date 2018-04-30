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

from __future__ import print_function

import argparse
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxboard import SummaryWriter

logging.basicConfig(level=logging.DEBUG)

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()

# define network

net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))


# data

def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    batch_size=opt.batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=False, transform=transformer),
    batch_size=opt.batch_size, shuffle=False)


def test(ctx):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def train(epochs, ctx):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.hybridize()

    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': opt.lr, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    param_names = params.keys()

    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    global_step = 0
    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            sw.add_scalar(tag='cross_entropy', value=L.mean().asscalar(), global_step=global_step)
            global_step += 1
            L.backward()

            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % opt.log_interval == 0 and i > 0:
                name, train_acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f' % (epoch, i, name, train_acc))

            # Log the first batch of images of each epoch
            if i == 0:
                sw.add_image('minist_first_minibatch', data.reshape((opt.batch_size, 1, 28, 28)), epoch)

        if epoch == 0:
            sw.add_graph(net)

        grads = [i.grad() for i in net.collect_params().values()]
        assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        for i, name in enumerate(param_names):
            sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)

        name, train_acc = metric.get()
        print('[Epoch %d] Training: %s=%f' % (epoch, name, train_acc))
        # logging training accuracy
        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc), global_step=epoch)

        name, val_acc = test(ctx)
        print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))
        # logging the validation accuracy
        sw.add_scalar(tag='accuracy_curves', value=('valid_acc', val_acc), global_step=epoch)

    sw.export_scalars('scalar_dict.json')
    sw.close()


if __name__ == '__main__':
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    train(opt.epochs, ctx)
    print('finished training')
