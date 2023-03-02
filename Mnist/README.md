# MNIST dataset

All notebooks in this folder train on MNIST hand-written dataset, a collection of black and white 28x28 images.
In order to keep it as simple and fast as possible, only the first 100 images are used (+ other 100 for testing)

### Neural Network Architecture

All notebooks share the same autoencoder architecture as well. It is a [784, 50, 2, 50, 784] fully connected with Tanh non-linearities. Moreover there is a l2 normalization layer before the latent space, such that the latent representation actually lay on the 1-D unit circle.

The ***encoder*** is defined as
```bash
encoder = nnj.Sequential(
                nnj.Flatten(),
                nnj.Linear(28*28, 50),
                nnj.Tanh(),
                nnj.Linear(50, 2),
                nnj.L2Norm(),
                add_hooks = True
        )
```
