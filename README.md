<div align="center">

# VAE-GAN <!-- omit in toc -->
[![ColabBadge]][notebook]

</div>

Digit-conditional VAE/GAN (Variational Autoencoder combined with a Generative Adversarial Network) on MNIST.

![Variational Autoencoder](https://i.ibb.co/Bsq0HjT/Screen-Shot-2019-05-28-at-10-44-45-AM.png)  
![VAE-GAN](https://i.ibb.co/1m6YHr1/Screen-Shot-2019-05-28-at-10-43-26-AM.png)  
![Example of turning inputs to 8's](https://i.ibb.co/RD86g3B/Screen-Shot-2019-05-28-at-11-04-59-AM.png)  

## Model
VAE-GAN = FC Encoder + NormDist + FC Decoder + FC Discriminator  

## Performance
- Train
  - 1h25min for 200 epoch (Google Colab NVIDIA T4, TF32-/AMP-)


[ColabBadge]:https://colab.research.google.com/assets/colab-badge.svg
[notebook]:https://colab.research.google.com/github/tarepan/VAE-GAN/blob/main/vaegan.ipynb
