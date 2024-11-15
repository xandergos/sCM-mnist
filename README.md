# sCM-mnist
#### [Paper](https://arxiv.org/abs/2410.11081)
Unofficial implementation of "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models" for MNIST

![The Algorithm](https://github.com/user-attachments/assets/a09d384c-e353-4466-8317-041a96d9d536)

This is a simple 2-file implementation of consistency training for MNIST. It can be easily adapted to consistency distillation in a few lines.
The code is intended to be short, and extremely easy to read and adapt to other use cases:

### train_consistency.py

Contains the core algorithm from the paper. The script will do consistency training by default, but if a 'model.pt' file exists, it will use it to do consistency distillation. 'model.pt' can be created with train_diffusion.py.

train.py uses torch.func.jvp, which calculates the JVP in the forward pass. Only some operations support JVP, but most common operations are supported. 
There are two exceptions: BatchNorm and Flash Attention. BatchNorm has some in-place operations that cause problems. The simple solution to this is to just avoid BatchNorm. I use GroupNorm in this implementation. Flash attention is more complicated, but the paper provides a way to calculate the JVP efficiently. I may implement this here in the future. Note that normal attention (see [here](https://github.com/NVlabs/edm/blob/main/training/networks.py) for one implementation) supports JVP.

### train_diffusion.py

This is a simple script to train a diffusion model with the TrigFlow parameterization. It creates 'model.pt' which can be distilled with train_consistency.py.

### unet.py

This is a simple UNet for MNIST. I ended up borrowing the UNet from [MNISTDiffusion](https://github.com/bot66/MNISTDiffusion).
Generally, the specifics of the model are not that important. However, many models may face issues with unstable gradients from time embeddings (see paper for details).
There are a few changes that are necessary to ensure that training is stable:
- Using Positional Embeddings instead of learned or fourier embeddings (Fourier embeddings with a small fourier scale can still work well)
- Using AdaGN *with PixelNorm* in order to inject time embeddings
- Added logvar output, which is used for adaptive loss weights
  
These changes are usually all that is necessary to ensure that training is stable for most models. 

#### Toy

There is also a folder called toy which contains some scripts for training a toy model. The script also outputs a plot of the model's gradients during training, which helps to visualize how consistency models work.
