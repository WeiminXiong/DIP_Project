import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag

def read_image(img_name):
    '''
    读取图片
    '''
    dir_path = '../training_data'
    path = os.path.join(dir_path, img_name)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def show_gray_image(image):
    '''
    显示灰度图片
    '''
    plt.imshow(image, cmap='gray')
    plt.show()

def show_color_image(image):
    '''
    显示彩色图片
    '''
    plt.imshow(image)
    plt.show()

def find_box_points(image):
    '''
    找到包围图像中白色像素的最小矩形
    return: 矩形角点，矫正所需顺时针旋转角度, 矫正所需顺时针旋转中心, 矩形长宽
    '''
    counters, hierarchy=cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_len = 0
    max_id = 0
    for i in range(len(counters)):
        if len(counters[i])>max_len:
            max_len = len(counters[i])
            max_id = i
    rect = cv2.minAreaRect(counters[max_id].squeeze())
    box = cv2.boxPoints(rect)
    edge_length = rect[1]
    if(edge_length[0]>edge_length[1]):
        angle = -(180 - rect[2])
    else:
        angle = -(90 - rect[2])
    return box, angle, rect[0], edge_length

def sort_box_points(box_points, center):
    '''
    将四个边界点按照（左下，左上，右上，右下）的顺序排列
    '''
    left_down=(0, 0)
    left_top=(0, 0)
    right_top=(0, 0)
    right_down=(0, 0)
    for point in box_points:
        if point[0]<center[0] and point[1]>center[1]:
            left_down = point
        if point[0]<center[0] and point[1]<center[1]:
            left_top = point
        if point[0]>center[0] and point[1]<center[1]:
            right_top = point
        if point[0]>center[0] and point[1]>center[1]:
            right_down = point
    return np.int0([left_down, left_top, right_top, right_down])

def rotate_and_crop_image(image, box, angle, center, edge_length):
    '''
    旋转裁剪图片至目标位置
    '''
    rotate_martrix = cv2.getRotationMatrix2D(center, angle, 1)
    width = max(edge_length)
    height = min(edge_length)
    # 将图片平移到中间
    rotate_martrix[0][2] += 500
    rotate_martrix[1][2] += 100
    # 旋转图片
    rotate_image = cv2.warpAffine(image, rotate_martrix, (2*int(width), 2*int(height)))
    # show_gray_image(rotate_image)
    # 算出旋转后的四个顶点
    pt1, pt2, pt3, pt4 = np.uint64(box)
    pt1 = np.dot(rotate_martrix, [[pt1[0]], [pt1[1]], [1]]).flatten()
    pt2 = np.dot(rotate_martrix, [[pt2[0]], [pt2[1]], [1]]).flatten()
    pt3 = np.dot(rotate_martrix, [[pt3[0]], [pt3[1]], [1]]).flatten()
    pt4 = np.dot(rotate_martrix, [[pt4[0]], [pt4[1]], [1]]).flatten()
    new_box = np.uint64([pt1, pt2, pt3, pt4])
    # print(new_box)
    # 裁剪的范围
    row_top = min(new_box[:,1])
    row_down = max(new_box[:, 1])
    col_left = min(new_box[:, 0])
    col_right = max(new_box[:, 0])
    cropped_image = rotate_image[row_top:row_down, col_left:col_right]
    return cropped_image

def morphology(image, mode, kernel_size):
    '''
    形态学变换
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if mode == 'dilate':
        image = cv2.dilate(image,kernel)
    if mode == "erode":
        image = cv2.erode(image, kernel)
    if mode == "open":
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if mode == "close":
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def correct_image(image):
    '''
    将图像颠倒的图片复原
    '''
    row, col = image.shape
    left_top = np.sum(image[0:int(0.4*row), 0:int(0.25*col)])
    right_down = np.sum(image[int(0.6*row):row, int(0.75*col):col])
    new_image = np.zeros_like(image)
    if(left_top<right_down):
        new_image = image[::-1, ::-1]
    else:
        new_image = image
    return new_image

def find_QRcode(image):
    '''
    找到图片中的二维码
    return: 二维码的边界坐标
    '''
    row, col = image.shape
    new_row = int(row*0.5)
    new_col = int(col*0.3)
    part_image = image[row-new_row:row, col-new_col: col]
    # show_gray_image(part_image)
    _, part_image = cv2.threshold(part_image, 20, 255, cv2.THRESH_BINARY)
    part_image = cv2.bitwise_not(part_image)
    morph_image = morphology(part_image, 'close', 45)
    morph_image = morphology(morph_image, 'open', 65)
    new_image = np.zeros_like(image)
    # 替换进同大小图像
    new_image[row-new_row:row, col-new_col:col] = morph_image
    # show_gray_image(new_image)
    box, angle, center, edge_length = find_box_points(new_image)
    return box, center

def clean_white_place(image, box, num):
    '''
    去除划分后图片中两侧空白的部分
    return: 更新后的box
    '''
    box = np.int0(box)
    gradient = (box[2][1]-box[1][1])*1.0/(box[2][0]-box[1][0])
    row_top = int(min(box[:, 1]))
    row_down = int(max(box[:, 1]))
    row_left = int(min(box[:, 0]))
    row_right = int(max(box[:, 0]))
    part_image = image[row_top: row_down, row_left: row_right]
    # show_gray_image(part_image)
    if num == 7:
        _, part_image = cv2.threshold(part_image, 140, 255, cv2.THRESH_BINARY_INV)
    if num ==21:
        _, part_image = cv2.threshold(part_image, 20, 255, cv2.THRESH_BINARY_INV)
    # show_gray_image(part_image)
    part_row, part_col = part_image.shape
    col_sum = np.sum(part_image, 0)
    left_padding = 0
    right_padding = 0
    for i in range(len(col_sum)):
        if col_sum[i]>254:
            break
        else:
            left_padding+=1
    for i in range(len(col_sum)):
        if col_sum[part_col-1-i]>254:
            break
        else:
            right_padding+=1
    # print(left_padding, right_padding)
    left_padding-=2
    left_padding = max(left_padding,0)
    right_padding-=2
    right_padding = max(right_padding, 0)
    # print(left_padding, right_padding)
    box[0,0]+=left_padding
    box[1,0]+=left_padding
    box[2,0]-=right_padding
    box[3,0]-=right_padding
    box[0,1] = box[0, 1]+left_padding*gradient
    box[1,1] = box[1, 1]+left_padding*gradient
    box[2,1] = box[2,1]-right_padding*gradient
    box[3,1] =box[3,1] -right_padding*gradient
    box = np.int0(box)
    # row_top = int(min(box[:, 1]))
    # row_down = int(max(box[:, 1]))
    # row_left = int(min(box[:, 0]))
    # row_right = int(max(box[:, 0]))
    # part_image = image[row_top: row_down, row_left: row_right]
    # show_gray_image(part_image)
    return box

def draw_21_code_line(image, gray_image,box, color = (255,0, 0), thickness = 1):
    '''
    在图中的21位码中画分割直线
    '''
    # 画出外围边框
    cv2.line(image, box[0], box[1], color, thickness)
    cv2.line(image, box[1], box[2], color, thickness)
    cv2.line(image, box[2], box[3], color, thickness)
    cv2.line(image, box[3], box[0], color, thickness)
    row, col, _ = image.shape
    image_copy = image.copy()
    gray_image_copy = gray_image.copy()
    row_top = min(box[:, 1])
    row_down = max(box[:, 1])
    col_left = min(box[:, 0])
    col_right = max(box[:, 0])
    # # print(box)
    # 画出内部分割线
    # gradient = (box[2, 1] - box[1,1])*1.0/(box[2, 0]-box[1, 0])
    # ratio = 1.52
    # col_distance = box[2, 0] - box[1, 0]
    # x1 = box[0, 0]
    # y1 = box[0, 1]
    # x2 = box[1, 0]
    # y2 = box[1, 1]
    # unit_distance = col_distance/(20+ratio)
    # # print(unit_distance)
    # for i in range(1, 21):
    #     if i<15:
    #         x3 = x2+i*unit_distance
    #         x4 = x1+i*unit_distance
    #     if i>=15:
    #         x3 = x2+(i-1+ratio)*unit_distance
    #         x4 = x1+(i-1+ratio)*unit_distance
    #     # print(x3, x4)
    #     y3 = y2+(x3-x2)*gradient
    #     y4 = y1+(x4-x1)*gradient
    #     x3 = int(x3)
    #     x4 = int(x4)
    #     y3 = int(y3)
    #     y4 = int(y4)
    #     cv2.line(image, (x3, y3), (x4, y4), color, thickness)
    mask_21 = np.ones((row, col), dtype=np.uint8)*255
    mask_21[row_top:row_down, col_left:col_right] = 0
    gray_image = cv2.bitwise_or(gray_image, mask_21)
    gray_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(gray_image, 2, 2)
    x_min = col_left
    x_list = []
    image_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x != 0 and y != 0 and w*h >= 80:
            x_list.append(x)
            if x < x_min:
                x_min = x
    x_list.sort()
    x_list.pop(0)
    line_num_21 = 0
    for i in range(len(x_list)):
        if line_num_21 == 14:
            dist_min = 20
        else:
            dist_min = 10
        if x_list[i] > x_min + dist_min:
            image_list.append(gray_image_copy[row_top: row_down, x_min: x_list[i]-1])
            cv2.line(image_copy, (x_list[i]-1, row_top), (x_list[i]-1, row_down), (255, 0, 0), 1)
            x_min = x_list[i]
            line_num_21 += 1
    image_list.append(gray_image_copy[row_top: row_down, x_min: col_right])
    # line_num_21 = 0
    if line_num_21 == 20:
        image[:, :, :] = image_copy[:, :, :]
    else:
        image_list = []
        interval = (col_right - col_left) / 21.5
        before_dist = 0
        dist = 0
        for i in range(20):
            if i == 14:
                dist += interval * 1.5
            else:
                dist += interval
            image_list.append(gray_image_copy[row_top: row_down, col_left+int(before_dist): col_left+int(dist)])
            cv2.line(image, (col_left+np.int0(dist), row_top), (col_left+np.int0(dist), row_down), (255, 0, 0), 1)
            before_dist = dist
    # for i in image_list:
    #     show_gray_image(i)
    return image_list
    # show_gray_image(image)
    # gray_image = np.add(gray_image,  mask_21 * 255)
    # show_gray_image(gray_image)
    # print(image.shape)
        

def draw_7_code_line(image, gray_image,box, color= (255, 0, 0), thickness = 1):
    '''
    在图中的7位码处画出分割线
    '''
    # 画出外围边框
    # print(box)
    cv2.line(image, box[0], box[1], color, thickness)
    cv2.line(image, box[1], box[2], color, thickness)
    cv2.line(image, box[2], box[3], color, thickness)
    cv2.line(image, box[3], box[0], color, thickness)
    row, col, _ = image.shape
    image_copy = image.copy()
    # show_color_image(image)
    row_top = min(box[:, 1])
    row_down = max(box[:, 1])
    col_left = min(box[:, 0])
    col_right = max(box[:, 0])
    col_distance = (col_right-col_left)
    # ratio = 1.5
    # x1 = box[0, 0]
    # y1 = box[0, 1]
    # x2 = box[1, 0]
    # y2 = box[1, 1]
    # gradient = (box[2,1]-box[1,1])*1.0/(box[2, 0]-box[1, 0])
    # unit_distance = col_distance/(6+ratio)
    # for i in range(1,7):
    #     x3 = x2 + (i-1+ratio)*unit_distance
    #     x4 = x1 + (i-1+ratio)*unit_distance
    #     y3 = y2 +(x3-x2)*gradient
    #     y4 = y1+(x4-x1)*gradient
    #     x3 =int(x3)
    #     x4 = int(x4)
    #     y3 = int(y3)
    #     y4 = int(y4)
    #     cv2.line(image, (x3, y3), (x4, y4), color, thickness)
    mask_7 = np.ones((row, col), dtype=np.uint8)*255
    mask_7[row_top:row_down, col_left:col_right] = 0
    gray_image = cv2.bitwise_or(gray_image, mask_7)
    gray_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)[1]
    # show_gray_image(gray_image)
    contours, hierarchy = cv2.findContours(gray_image, 2, 2)
    x_min = col_left
    x_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x != 0 and y != 0 and w*h >= 100:
            x_list.append(x)
            if x < x_min:
                x_min = x
    x_list.sort()
    x_list.pop(0)
    line_num_7 = 0
    for i in range(len(x_list)):
        if line_num_7 == 0:
            dist_min = 20
        else:
            dist_min = 10
        if x_list[i] > x_min + dist_min:
            cv2.line(image_copy, (x_list[i]-3, row_top), (x_list[i]-3, row_down), (255, 0, 0), 1)
            x_min = x_list[i]
            line_num_7 += 1
    # line_num_21 = 0
    # line_num_7 =6
    # print(line_num_7)
    if line_num_7 == 6:
        image[:, :, :] = image_copy[:, :, :]
    else:
        interval = (col_right - col_left) / 7.4
        dist = 0
        for i in range(6):
            if i == 0:
                dist += interval * 1.4
            else:
                dist += interval
            cv2.line(image, (col_left+np.int0(dist), row_top), (col_left+np.int0(dist), row_down), (255, 0, 0), 1)


def find_num_code(image, color_image):
    '''
    找到图中的21位码，并画出7位码和21位码边框
    return: 切割好的图片
    '''
    QR_box, QR_center = find_QRcode(image)
    QR_box = sort_box_points(QR_box, QR_center)
    # print(QR_box)
    # 二维码方框对应顶点
    QR_row_top = int(min(QR_box[:, 1]))
    QR_row_down = int(max(QR_box[:, 1]))
    QR_col_left = int(min(QR_box[:, 0]))
    QR_col_right = int(max(QR_box[:, 0]))
    QR_row_distance = QR_row_down - QR_row_top
    # print(QR_row_distance)
    # 21位码对应顶点（粗划分）
    num21_row_top = QR_row_down + int(0.3/4.5*QR_row_distance)
    num21_row_down = QR_row_down + int(1.6/4.5*QR_row_distance)
    num21_col_right = QR_col_left - int(9.3/4.5*QR_row_distance)
    num21_col_left = QR_col_left - int(22/4.5*QR_row_distance)
    num21_col_left = max(0, num21_col_left)
    # print(num21_row_top, num21_row_down, num21_col_left, num21_col_right)
    num21_image = image[num21_row_top:num21_row_down, num21_col_left:num21_col_right]
    # 21位码对应顶点（细化分）
    # show_gray_image(num21_image)
    thresh, binarized_image = cv2.threshold(num21_image, 20, 255, cv2.THRESH_BINARY)
    binarized_image = cv2.bitwise_not(binarized_image)
    morph_image = morphology(binarized_image, 'close', 21)
    morph_image = morphology(morph_image, 'dilate', 9)
    # show_gray_image(morph_image)
    new_image = np.zeros_like(image)
    new_image[num21_row_top:num21_row_down, num21_col_left:num21_col_right] = morph_image
    box, num21_angle, center, num21_edge_length = find_box_points(new_image)
    box = sort_box_points(box, center)
    num21_cleaned_box = clean_white_place(image, box, 21)
    image_list = draw_21_code_line(color_image, image, num21_cleaned_box)
    # show_color_image(color_image)
    # 7位码对应顶点（粗划分）
    num7_row_top = QR_row_top - int(10.7/4.5*QR_row_distance)
    num7_row_down = QR_row_top - int(7.3/4.5*QR_row_distance)
    num7_col_right = QR_col_left - int(13/4.5*QR_row_distance)
    num7_col_left = QR_col_left -int(22.2/4.5*QR_row_distance)
    if num7_row_top<0:
        num7_row_top=10
    if num7_col_left<0:
        num7_col_left =0
    num7_image = image[num7_row_top:num7_row_down, num7_col_left:num7_col_right]
    # 形态学处理
    binarized_image = np.uint8((num7_image> 60)&(num7_image<140))
    binarized_image*=255
    # show_gray_image(binarized_image)
    num7_morph_image = morphology(binarized_image, 'open', 2)
    num7_morph_image = morphology(num7_morph_image, 'close', 3)
    num7_median_image = cv2.medianBlur(num7_morph_image, 5)
    num7_median_image = cv2.medianBlur(num7_median_image, 3)
    # retval, labels, stats, _ = cv2.connectedComponentsWithStats(num7_median_image, connectivity=8)
    # for i in range(retval):
    #     if stats[i][4] < 150:
    #         num7_median_image[labels==i] = 0
    # show_gray_image(num7_median_image)
    # 7位码对应顶点（细划分）
    num7_rect_image = morphology(num7_median_image, 'close', 40)
    num7_rect_image = morphology(num7_rect_image, 'dilate', 21)
    # show_gray_image(num7_rect_image)
    new_image = np.zeros_like(image)
    new_image[num7_row_top: num7_row_down, num7_col_left:num7_col_right] = num7_rect_image
    box, angle, center, edge_length = find_box_points(new_image)
    box = sort_box_points(box, center)
    cleaned_box = clean_white_place(image, box, 7)
    # show_gray_image(num7_rect_image)
    draw_7_code_line(color_image, image, cleaned_box)
    return image_list

# def cut_num21_code(image, box, angle, edge_length):
#     '''
#     切割21位码， 用于后续的识别
#     return: 切割好的图片
#     '''
#     x_average = int(np.mean(box[:, 0]))
#     y_average = int(np.mean(box[:, 1]))
#     center = (x_average, y_average)
#     if angle<-90:
#         angle+=180
#     num21_image = rotate_and_crop_image(image, box, angle, center, edge_length)
#     # show_gray_image(num21_image)
#     # thresh, num21_image = cv2.threshold(num21_image, 70, 255, cv2.THRESH_BINARY)
#     image_list = []
#     row, col = num21_image.shape
#     ratio = 1.52
#     unit_distance = col/(20+ratio)
#     x = 0
#     for i in range(1, 22):
#         if i<15:
#             x_new = i*unit_distance
#         if i>=15:
#             x_new = (i-1+ratio)*unit_distance
#         # print(x3, x4)
#         x_new = int(x_new)
#         image_list.append(num21_image[:, x:x_new])
#         x = x_new
#     # show_gray_image(num21_image)
#     # for i in range(21):
#     #     show_gray_image(image_list[i])
#     return image_list


def premanage(image):
    '''
    预处理输入的图片
    '''
    _,morph_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    morph_image = cv2.medianBlur(morph_image, 5)
    morph_image = morphology(morph_image, 'close', 20)
    morph_image = morphology(morph_image, 'open', 80)
    box, angle,center, edge_length = find_box_points(morph_image)
    # print(box, angle,center, edge_length)
    rotate_image = rotate_and_crop_image(image, box, angle, center, edge_length)
    # show_gray_image(morph_image)
    rotate_image = correct_image(rotate_image)
    # show_gray_image(rotate_image)
    return rotate_image


if __name__ == '__main__':
    image = read_image('2018-5-22-17-58-10.bmp')
    _,morph_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    morph_image = cv2.medianBlur(morph_image, 5)
    morph_image = morphology(morph_image, 'close', 20)
    morph_image = morphology(morph_image, 'open', 80)
    box, angle,center, edge_length = find_box_points(morph_image)
    # print(box, angle,center, edge_length)
    rotate_image = rotate_and_crop_image(image, box, angle, center, edge_length)
    # show_gray_image(morph_image)
    rotate_image = correct_image(rotate_image)
    # show_gray_image(rotate_image)
    color_image = cv2.cvtColor(rotate_image, cv2.COLOR_GRAY2RGB)
    image_list= find_num_code(rotate_image, color_image)
    # print(num21_cleaned_box)
    # show_gray_image(rotate_image)
    # image_list = cut_num21_code(rotate_image, num21_cleaned_box, num21_angle, edge_length)
    # for i in range(21):
    #     show_gray_image(image_list[i])
    show_color_image(color_image)