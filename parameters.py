using_dot_generator = False


# ################## datas parameters (img size, color, etc)  ######################
n = 8
img_size = 256
img_as_gray = False

vae = False

# ############### data settings: type of data, folder, library, etc ###############
dataset_folder = 'D:/DataSets/dogs_cats'
dataset_ToSplit_folder = 'D:/DataSets/dogs_cats'
destin_splitted_folder = 'D:/DataSets/splitted/dogs_cats'

# #################### if test_split = 0, then will be only validation and test ##################
# if u need split data on 3 categories, then set non zero values here
# but if u already split the data on 2 categories and want to resplit it into 3, u should delete earlier splitted data
clear_splitted_data = False
split_data = True
validation_split = 0.2
test_split = 0.1

mnist_d = False
use_flow_from_directory = True
use_flow = True
use_my_pd = False  # using my preprocessing data algorithm

# ##################### training parameters for models ############################
batch_size = 100
latent_dim = 128  # to be easier generate and visualize result
dropout_r = 0.1
lr_0 = 0.0001
epoch = 50

