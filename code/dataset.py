import os

dir_path = '../dataset/number'
name_list = os.listdir(dir_path)
train_path = dir_path+'/train'
valid_path = dir_path+'/valid'
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(valid_path):
    os.makedirs(valid_path)
for i in name_list:
    dir_name = os.path.join(dir_path, i)
    file_name_list = os.listdir(dir_name)
    length = len(file_name_list)
    new_train_dir_name = os.path.join(train_path, i)
    if not os.path.exists(new_train_dir_name):
        os.makedirs(new_train_dir_name)
    new_valid_dir_name = os.path.join(valid_path, i)
    if not os.path.exists(new_valid_dir_name):
        os.makedirs(new_valid_dir_name)
    for j in range(length):
        file_name = os.path.join(dir_name, file_name_list[j])
        if j%5==0:
            new_file_name = os.path.join(new_valid_dir_name, file_name_list[j])
        else:
            new_file_name = os.path.join(new_train_dir_name, file_name_list[j])
        os.rename(file_name, new_file_name)