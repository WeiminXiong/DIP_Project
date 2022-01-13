import os
import cv2
import utils

directory_path = '../training_data'
annotation_path = os.path.join(directory_path, 'annotation.txt')
letter_dir = '../dataset/letter'
number_dir = '../dataset/number'

def resize_img_keep_ratio(img,target_size):
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new

if __name__ == '__main__':
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
        if not os.path.exists(number_dir):
            os.makedirs(number_dir)
        for line in lines:
            file_name, tag_21, tag_7 = line.split()
            image_name = file_name.split('.')[0]
            image = utils.read_image(file_name)
            rotate_image = utils.premanage(image)
            color_image = cv2.cvtColor(rotate_image, cv2.COLOR_GRAY2RGB)
            image_list = utils.find_num_code(rotate_image, color_image)
            # image_list = utils.cut_num21_code(rotate_image, num21_cleaned_box, num21_angle, rotate_image.shape)
            # utils.show_gray_image(resize_img_keep_ratio(image_list[0], [50, 50]))
            i = 0
            for image, tag in zip(image_list, tag_21):
                i+=1
                _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
                if not i == 15:
                    dir_path = os.path.join(number_dir, tag)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    image_save_path = os.path.join(dir_path, image_name+'_'+str(i)+'.jpg')
                    resized_image = resize_img_keep_ratio(image, (64,64))
                    cv2.imwrite(image_save_path, resized_image)
                else:
                    dir_path = os.path.join(letter_dir, tag)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    image_save_path = os.path.join(dir_path, image_name+'_'+str(i)+'.jpg')
                    resized_image = resize_img_keep_ratio(image, (64, 64))
                    cv2.imwrite(image_save_path, resized_image)