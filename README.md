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

### VariationalAutoencoder.py and VariationalAutoencoder.py
Simple examples of VAE (with only dense layers). <br>
First one uses mnist dataset.<br>
Second one is attempt to load pets data and see how the output will look like.<br>
Now the output looks like this:<br>
![input_output_50ep_ims112_simpleVAE](https://user-images.githubusercontent.com/60517813/174012570-e5188dfb-c5c1-493e-b257-1775961837c4.jpg)
<br>Here VAE is very simple, so it cannt give good output.

### Further
I have plans to test different data preprocessing methods, write more complicated NN for encoder and decoder in AE to improve quality of output data. 
