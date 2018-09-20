# BourGAN: Generative Networks with Metric Embeddings (Deverloper version)

## Abstract:
This paper addresses the mode collapse for generative adversarial networks (GANs). We view modes as a geometric structure of data distribution in a metric space. Under this geometric lens, we embed subsamples of the dataset from an arbitrary metric space into the l2 space, while preserving their pairwise distance distribution. Not only does this metric embedding determine the dimensionality of the latent space automatically, it also enables us to construct a mixture of Gaussians to draw latent space random vectors. We use the Gaussian mixture model in tandem with a simple augmentation of the objective function to train GANs. Every major step of our method is supported by theoretical analysis, and our experiments on real and synthetic data confirm that the generator is able to produce samples spreading over most of the modes while avoiding unwanted samples, outperforming several recent GAN variants on a number of metrics and offering new features.



## Publication  
Chang Xiao, Peilin Zhong, and Changxi Zheng. "BourGAN: Generative Networks with Metric Embeddings." Neural Information processing systems (NIPS), 2018. (**Spotlight Presentation**) [[PDF](https://arxiv.org/abs/1805.07674)] [[bibtex](https://raw.githubusercontent.com/a554b554/BourGAN/master/bibtex)]


## Requirements
pytorch 0.4.1

numpy 1.14.3

scipy 1.1.0

matplotlib 2.2.3

#
The repo is still under development and will update frequently.