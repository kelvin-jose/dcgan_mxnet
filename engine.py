import config
import mxnet as mx
from mxnet import nd
from mxnet import random
from mxnet import autograd


def train_generator(g_model, d_model, g_trainer, device, loss_fn):
    noise = random.randn(config.BATCH_SIZE, 100, 1, 1, ctx=device)
    labels = nd.ones(shape=(1, config.BATCH_SIZE), ctx=device)

    with autograd.record():
        g_output = g_model(noise)
        d_output = d_model(g_output)
        loss = loss_fn(labels, d_output.reshape(1, -1))

    loss.backward()
    loss = loss.copyto(mx.cpu())
    g_trainer.step(config.BATCH_SIZE)
    return loss


def train_discriminator(images, g_model, d_model, d_trainer, device, loss_fn):
    images = images.copyto(device)
    labels = nd.ones(shape=(1, config.BATCH_SIZE), ctx=device)
    fake_labels = nd.zeros(shape=(1, config.BATCH_SIZE), ctx=device)
    noise = random.randn(config.BATCH_SIZE, 100, 1, 1, ctx=device)

    with autograd.record():
        output_0 = d_model(images)
        loss_0 = loss_fn(output_0.reshape(1, -1), labels)
        g_output = g_model(noise)
        d_output = d_model(g_output)
        loss_1 = loss_fn(fake_labels, d_output.reshape(1, -1))
        loss = loss_0 + loss_1

    loss.backward()
    loss_0 = loss_0.copyto(mx.cpu())
    loss_1 = loss_1.copyto(mx.cpu())
    d_trainer.step(config.BATCH_SIZE)
    return loss_0, loss_1