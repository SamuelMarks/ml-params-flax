# Originally taken from https://github.com/google/flax/blob/46960d9/examples/mnist/train.py
#
# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

This script trains a simple Convolutional Neural Net on the MNIST dataset.
The data is loaded using tensorflow_datasets.

"""
from sys import modules

import jax
import jax.numpy as jnp
import numpy as onp
from flax import nn
from jax import random

from ml_params_flax import get_logger

logging = get_logger(':'.join(('ml_params_flax', modules[__name__].__name__)))


class CNN(nn.Module):
    """A simple CNN model."""

    def apply(self, x):
        x = nn.Conv(x, features=32, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(x, features=64, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=10)
        x = nn.log_softmax(x)
        return x


def onehot(labels, num_classes=10):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""

    def loss_fn(model):
        logits = model(batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics


@jax.jit
def eval_step(model, batch):
    logits = model(batch['image'])
    return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm] for k, v in train_ds.items()}
        optimizer, metrics = train_step(optimizer, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: onp.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                 epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

    return optimizer, epoch_metrics_np


def eval_model(model, test_ds):
    metrics = eval_step(model, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']
