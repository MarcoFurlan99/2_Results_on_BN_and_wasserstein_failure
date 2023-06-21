# Result 1: the failure of the feature layer

After many failed attempts to make the Wasserstein distance work, I decided to check for normality. Doing so I put my hands on the feature tensor from the last downsample step (which has size (batch_size x 4 x 4 x 1024), since my starting images are 64x64). Specifically, I took the models trained on the following datasets (5000 training samples per dataset):

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/samples.png?raw=true)

So as I said I was checking for gaussianity. We are of course talking about a 1024-dimensionality distribution, whose gaussianity is not easy to check, but a good start is to check if the 1-d projections are gaussian. Consequently, I took the 5000 4x4x1024 latent spaces (one per image), squeezed them on a 80000x1024 matrix, and created for each column of such matrix (that is, for each dimension of the probability space) a histogram with all the samples, totalling 1024 histograms with 80000 samples each.

Firstly, I show some of these histograms for $\mu_2 - \mu_1 = 10$:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_10.png?raw=true)

So this brings up naturally a bunch of observations:

- There are a lot of dimensions which collapse (partially or entirely) to 0, because of the ReLU preceding the latent space

- the gaussianity assumption does not hold. Not only for the obvious zero-dimensions, which are not a big bother since we can just remove them (a simple torch.min(dimension) == 0 would work), but the issue is that most dimensions show a more various behaviour then a simple 1-d gaussian. Remembering that these are projections we conclude that that 1024-d distribution has a behaviour which is much more complex that a Gaussian, and should not (in my opinion) modeled by that.

Despite all of this, we can still try the Wasserstein after removing the "zero-dimensions". And I would've done so, if it wasn't for the results I got checking the latent space of the model trained on $\mu_2 - \mu_1 = 22$. Here is what I got:


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_22.png?raw=true)

So that's confusing... I consequently had to filter for dimensions which did not just contain zeros (I did more generally torch.min(dimension) != 0) and I got:


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_22_nonzero.png?raw=true)

So.. just 3 to 8 values per dimension, of order 1e-15? (Is it? I don't even know what that 1e-15+...e-8 scale is tbh). Weird, so I checked the latent space of the model trained on $\mu_2 - \mu_1 = 254$, which is the best performing on its own dataset, here are the results (after removing "zero-dimensions"):

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_254_nonzero.png?raw=true)

So again, a bunch of numbers instead of a ditribution. What is even MORE notable, and it is not evident from this last graph (because I rescaled) but can be seen from the graph above, is that values sum up to multiples of 5000. It came naturally to do a per-image check. So I chose the dimension corresponding to the bottom-right graph, and plotted the 5000 histograms, one per image, summing the 16 numbers of the feature space.

Here are the first 16:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/mu_distance_254_per_image.png?raw=true)

As you can image, also the remaining 1484 looked the same.

In conclusion, the latent space is the same no matter the input image. So why is that? My personal concolusion is that the task of classification in the cases $\mu_2 - \mu_1 = 22$ and $\mu_2 - \mu_1 = 254$ is "too simple", to the point that the use of the last latent space is not required, and the entirety of the learning is done on the other layers. This is possible thanks to the structure of the U-Net, which includes "copy and crop" passages from the contracting path to the expansive path.

So if we want to compute a distance between these latent spaces we should take into account the fact that the values in such latent space may be, in some cases, just a few numbers very close to zero, at least for the source-to-original network case, which in my opinion highly discourages the possibility of any distance concerning this layer. I would instead try to focus on higher feature spaces (or even on the images themselves? which would be absolutely ridicolous honestly but it is also a possibility).

# Result 2: the success of BN adaptation

I tried changing around some parameters for the usual IoU graph procedure and got very interesting results. So this time i fixed:

- $\sigma_1 = \sigma_2 = 50$

- $\mu_2 - \mu_1 = k$ where $k$ is constant

I then did the computation changing $\mu_1$ among the different datasets, and did it three times for $k = 20, 40, 80$. These are the resulting samples (I know image is small but it is important to see how they are, click on it and zoom):

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/samples_all.png?raw=true)

And here are the usual graphs:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/graph_the_3_musketeers_3.png?raw=true)


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/graph_the_3_musketeers_4.png?raw=true)


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/graph_the_3_musketeers_5.png?raw=true)

So great results in favour of BN! I also wanted to include some extra graphs that show the problem in the U-Net WITHOUT batch norm graph. Images correspond to the predictions:


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/source_models_predictions_3.png?raw=true)

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/source_models_predictions_4.png?raw=true)

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/source_models_predictions_5.png?raw=true)

It is not necessary to see the ground truth masks to understand what is happening (see above samples). Whiter images are all interpreted as mask, and blacker images are all interpreted as not-mask. Hence the values 0.20, 0.21 (average area occupied by masks) and 0.0 (no intersection nor union --> IoU = 0).

I also include the training history because it is unusually messy:


  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/training_history_3.png?raw=true)

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/training_history_4.png?raw=true)

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/BN_results/training_history_5.png?raw=true)
