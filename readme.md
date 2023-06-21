# Result 1: the failure of the feature layer

After many failed attempts to make the Wasserstein distance work, I decided to check for normality. Doing so I put my hands on the feature tensor from the last downsample step (which has size (batch_size x 4 x 4 x 1024), since my starting images are 64x64). Specifically, I took the models trained on the following datasets (5000 training samples per dataset):

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/samples.png?raw=true)

So as I said I was checking for gaussianity. We are of course talking about a 1024-dimensionality distribution, whose gaussianity is not easy to check, but a good start is to check if the 1-d projections are gaussian. Consequently, I took the 5000 4x4x1024 latent spaces (one per image), squeezed them on a 80000x1024 matrix, and created for each column of such matrix (that is, for each dimension of the probability space) a histogram with all the samples, totalling 1024 histograms with 80000 samples each.

Firstly, I show some of these histograms for $\mu_2 - \mu_1 = 10$:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_10.png?raw=true)

So this brings up naturally a bunch of observations:

- There are a lot of dimensions which collapse (partially or entirely) to 0, because of the ReLU preceding the latent space

- the gaussianity assumption does not hold. Not only for the obvious zero-dimensions, which are not a big bother since we can just remove them (a simple torch.min(dimension) == 0 would work), but the issue is that most dimensions show a more various behaviour then a simple 1-d gaussian. Remembering that these are projections we conclude that that 1024-d distribution has a behaviour which is much more complex that a Gaussian, and should not (in my opinion) modeled by that.

Despite all of this, we can ideally still try the Wasserstein. And I would've done so, if it wasn't for the results I got checking the latent space for $\mu_2 - \mu_1 = 22$. Here is what I got:


# Result 2: the success of BN adaptation

