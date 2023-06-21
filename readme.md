# Result 1: the failure of the feature layer

After many failed attempts to make the Wasserstein distance work, I decided to check for normality. Doing so I put my hands on the output tensor from the last downsample step (which has size (batch_size x 4 x 4 x 1024), since my starting images are 64x64). Specifically, I took the models trained on the following datasets:

  ![alt text](https://github.com/MarcoFurlan99/2_Results_on_BN_and_Wasserstein_failure/blob/master/feature_space/samples.png?raw=true)

# Result 2: the success of BN adaptation