
from train import ThreeLayerConvNet
import cv2
import os
import utils
import torch
import numpy as np
from segment import resize_img_keep_ratio

annotation_path = '../training_data/annotation.txt'
segments_path = '../segments'
prediction_path = '../prediction.txt'
if not os.path.exists(segments_path):
    os.makedirs(segments_path)


number_path = '../model_number.pth'
letter_path = '../model_letter.pth'
number_model = ThreeLayerConvNet(3, 32, 32, 10)
letter_model = ThreeLayerConvNet(3, 32, 32, 26)
number_model.load_state_dict(torch.load(number_path))
letter_model.load_state_dict(torch.load(letter_path))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
number_model.to(device)
letter_model.to(device)


with open(annotation_path, 'r', encoding='utf-8') as f:
    w = open(prediction_path, 'w', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        file_name, _, _ = line.split()
        image = utils.read_image(file_name)
        rotate_image = utils.premanage(image)
        color_image = cv2.cvtColor(rotate_image, cv2.COLOR_GRAY2RGB)
        image_list = utils.find_num_code(rotate_image, color_image)
        number_list = []
        letter_list  = []
        idx = 0
        for item in image_list:
            idx+=1
            if idx !=15:
                item = resize_img_keep_ratio(cv2.cvtColor(item, cv2.COLOR_GRAY2RGB), (64,64))
                number_list.append(torch.tensor(item))
            else:
                item = resize_img_keep_ratio(cv2.cvtColor(item, cv2.COLOR_GRAY2RGB), (64,64))
                letter_list.append(torch.tensor(item))
        number_tensor = torch.stack(number_list).float()
        letter_tensor = torch.stack(letter_list).float()
        number_tensor = number_tensor.permute([0,3,1,2])
        letter_tensor = letter_tensor.permute([0,3,1,2])
        # print(letter_tensor.shape)

        _, letter_preds = letter_model(letter_tensor.to(device)).max(1)
        _, number_preds = number_model(number_tensor.to(device)).max(1)
        number_preds = number_preds.tolist()
        letter_preds = letter_preds.item()
        for i in range(20):
            number_preds[i] = chr(ord('0')+number_preds[i])
        num7_result = ""
        num21_result = ""
        for i in range(14):
            num21_result+=number_preds[i]
        num21_result+=chr(ord('A')+letter_preds)
        num7_result+=chr(ord('A')+letter_preds)
        for i in range(6):
            num7_result+=number_preds[i+14]
            num21_result+=number_preds[i+14]
        save_path = os.path.join(segments_path, file_name)
        cv2.imwrite(save_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        w.write(file_name+' '+num21_result+' '+num7_result+'\n')
    w.close()