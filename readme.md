# Result 1: the failure of the feature layer

After many failed attempts to make the Wasserstein distance work, I decided to check for normality. Doing so I put my hands on the output tensor from the last downsample step (which has size (batch_size x 4 x 4 x 1024), since my starting images are 64x64). Specifically, I took the models trained on the following datasets (5000 training samples per dataset):

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/samples.png?raw=true)

So as I said I was checking for gaussianity. We are of course talking about a 1024-dimensionality distribution, whose gaussianity is not easy to check, but a good start is to check if the 1-d projections are gaussian. Consequently, I took the 5000 4x4x1024 latent spaces (one per image), squeezed them on a 80000x1024 matrix, and created for each column of such matrix (that is, for each dimension of the probability space) a histogram with all the samples, totalling 1024 histograms with 80000 samples each.

Firstly, I show some of these histograms for $\mu_2 - \mu_1 = 10$:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_10.png?raw=true)


# Result 2: the success of BN adaptation

