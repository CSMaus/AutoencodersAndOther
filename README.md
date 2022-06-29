### Here autoencoders and data preprocessing for AEs
### All scripts are in draft form. Here I'm just learning and posting my attempts to solve problems in different ways!

# Data preprocessing scripts

## SplitDataset script
This script is to split dataset into *train*, *validation* and *test* datasets.<br>

**Pay Attention!!!** It's only copyes data into destination folder without removing data from initital dataset folder.<br>

You can set dataset folder and destination folder in **parameters.py** script. Script will define categories and split data with saving categorization.
If you set *validation_split* or *test_split* value to 0, then script will not create such a folder with data.<br>
You alse can clear created splitted data if you set **clear_splitted_data** variable to **True** value.<br>

## Preprocess data
I want to use this script to test different ways to load and preprocess image data.<br>

# Autoencoders
Now I'm just studying autoencoders, so here will be a lot of attempts of making different autoencoders.<br>

### mnistAE.py and ManifoldLearning.py
Here there are few simple autoencoders that uses mnist dataset. It's just to understand how does AE works.<br>

### VariationalAutoencoderPets.py and VariationalAutoencoder.py
Simple examples of VAE and conditional VAE (CVAE) (with only dense layers). <br>
VariationalAutoencoder.py uses mnist dataset. Now I remada it into CVAE.<br>
VariationalAutoencoderPets.py is attempt to load pets data and see how the output will look like.<br>
Now the output looks like this:<br>
![input_output_50ep_ims112_simpleVAE](https://user-images.githubusercontent.com/60517813/174012570-e5188dfb-c5c1-493e-b257-1775961837c4.jpg)
<br>Here VAE is very simple, so it cannt give good output.<br>

### ConditionalVAE_space_sett.py
Here first very simple conditional VAE thst uses pets images (it uses labels also to improve accuracy). Output quality is velry low and bad, but results much better than in VAE.<br>
Such low quality results are due to the too simple NN architecure (3 Fully connetcted layers) that is the base for this AE.<br>
![ConditioinalVAE_2Dense_lat_dim128_ims128](https://user-images.githubusercontent.com/60517813/175881094-e08b50f1-cb3b-4351-b312-95c1467aeb69.jpg)
<br>

### ConditionalVAE_space_sett.py
Here convolutional NN in encoder/decoder.<br>
![cCVAE_conv_ims280_latdim128](https://user-images.githubusercontent.com/60517813/176413925-74c3ef08-7ed9-4231-a014-71c6dfa696ed.png)
<br>

### Further
I have plans to upgrade and test different data preprocessing methods, write more complicated NN for encoder and decoder in AE to improve quality of output data.<br>
But the next step (and 2 scripts) will be for the GAN (simple GAN with Conv NN as base).
