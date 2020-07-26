import config
from model import Generator
from model import Discriminator
from mxnet import npx
from mxnet import gluon
from mxnet import init
from tqdm import tqdm
import mxnet as mx
import numpy as np
from mxnet.optimizer import Adam
from mxnet.gluon.data import DataLoader
from mxnet.gluon.loss import SigmoidBCELoss
from engine import train_generator
from engine import train_discriminator


device = npx.gpu() if npx.num_gpus() > 0 else npx.cpu()

gen = Generator()
gen.collect_params().initialize(init=init.Normal(sigma=0.02), force_reinit=True, ctx=device)
# noise = random.randn(1, 100, 1, 1)
# output = gen(noise)
# print(output.shape)

dis = Discriminator()
dis.collect_params().initialize(init=init.Normal(sigma=0.02), force_reinit=True, ctx=device)
# noise = random.randn(1, 3, 64, 64)
# output = dis(noise)
# print(output.shape)

loss_fn = SigmoidBCELoss()

dataset = mx.gluon.data.vision.datasets.ImageFolderDataset('input/')
transform_fn = mx.gluon.data.vision.transforms.Compose([
    mx.gluon.data.vision.transforms.Resize(size=(64, 64)),
    mx.gluon.data.vision.transforms.ToTensor(),
    mx.gluon.data.vision.transforms.Normalize(mean=[0.49139969, 0.48215842, 0.44653093],
                                              std=[0.20220212, 0.19931542, 0.20086347]),
])
dataset_transformed = dataset.transform_first(transform_fn)

generator_trainer = gluon.Trainer(dis.collect_params(), 'adam', {'learning_rate': 0.0002, 'beta1': 0.5})
discriminator_trainer = gluon.Trainer(dis.collect_params(), 'adam', {'learning_rate': 0.0002, 'beta1': 0.5})

train_dataloader = DataLoader(dataset_transformed, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

for epoch in range(config.EPOCHS):
    real_losses, fake_losses, gen_losses = [], [], []
    for batch_n, (images, _) in tqdm(enumerate(train_dataloader)):
        real_loss, fake_loss = train_discriminator(images, gen, dis, discriminator_trainer, device, loss_fn)
        gen_loss = train_generator(gen, dis, generator_trainer, device, loss_fn)
        real_losses.append(real_loss)
        fake_losses.append(fake_loss)
        gen_losses.append(gen_loss)
        if batch_n % 50 == 0:
            print(f"epoch # {epoch} batch # {batch_n} real loss: {real_loss.asscalar():.4f} fake loss: {fake_loss.asscalar():.4f} generator loss: {gen_loss.asscalar():.4f}")




