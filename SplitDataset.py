# this script i'm gonna use to split dataset into training, validation and test
import parameters as p, numpy as np, sys, os
from tqdm import tqdm
import shutil


# here I'm just copying files
# it's possible to replace files from "datas_folder" to "destin_folder"
# but to the moment I need just copy files, so it will not be random

datas_folder = p.dataset_ToSplit_folder
destin_folder = f'{p.destin_splitted_folder}'
v_s = p.validation_split
t_s = p.test_split

path = os.path.join(datas_folder)
path_list = os.listdir(path)
print('Number of categories', np.shape(path_list)[0])
print('Categories is:', path_list)

if p.clear_splitted_data:
    if os.path.exists(destin_folder):
        tqdm(shutil.rmtree(destin_folder, ignore_errors=True))
    print('Data deleted')

if p.split_data:
    for category in path_list:
        imgs = os.listdir(os.path.join(datas_folder, category))
        num_imgs = len(imgs)
        print(f'\nTotal number of images in category {category}:', num_imgs)

        i = 0
        for img in tqdm(imgs):

            if i < int(num_imgs * (1 - v_s - t_s)):
                train_val_folder = 'train'
            elif int(num_imgs * (1 - v_s - t_s)) <= i < int(num_imgs * (1 - t_s)):
                train_val_folder = 'validation'
            else:
                train_val_folder = 'test'
            i += 1
            destin = f'{destin_folder}/{train_val_folder}/{category}'
            if os.path.exists(destin):
                while not os.path.exists(f'{destin}/{img}'):
                    shutil.copyfile(f'{datas_folder}/{category}/{img}', f'{destin}/{img}')
            else:
                os.makedirs(destin)
                while not os.path.exists(f'{destin}/{img}'):
                    shutil.copyfile(f'{datas_folder}/{category}/{img}', f'{destin}/{img}')
            # break















