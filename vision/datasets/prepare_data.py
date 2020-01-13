import os
import cv2
import numpy as np
import pickle
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
import ipdb
from vision.datasets.score_plate import ScorePlate

max_samples = 30000
cache_dir = '/tmp/voc_cache'
# def img_open_filter(img):
#     """
#     图像开运算，去除噪点类干扰以及补全验证码缺口
#     :param img:
#     :return:
#     """
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
#     newImg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     newImg = cv2.dilate(newImg, kernel2)
#     return newImg

# def img_close_filter(img):
#     """
#     图像闭运算，去除干扰
#     :param img:
#     :return:
#     """
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
#     closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#     newImg = cv2.erode(closed, kernel2, iterations=1)
#     return newImg
def gen_score_plate_data():
    debug_vis = False
    os.makedirs(cache_dir, exist_ok= True)
    plate = ScorePlate()
    all_annots = []
    for idx in tqdm(range(max_samples)):
        m_up = int(np.random.uniform(0, 99))
        s_up = int(np.random.uniform(0, 59))
        ms_up = int(np.random.uniform(0, 59))
        # ipdb.set_trace()
        img, annot_list = plate.render_time(m_up, s_up, ms_up)
        if not debug_vis:
            output_name = os.path.join(cache_dir, '{}.jpg'.format(idx))
            all_annots.append((output_name, annot_list))
            cv2.imwrite(output_name, img)
        else:
            plate.render_annots(img, annot_list)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    if not debug_vis:
        out_name = os.path.join(cache_dir, 'pick.pk') 
        with open(out_name, 'wb')  as fid:
            pickle.dump(all_annots, fid)
        print(f'save annot data to {cache_dir}')

def gen_voc_dataset():
    from vision.datasets.voc_writer import convertToVOCFormat
    voc_output_dir = '../../Dataset'
    out_name = os.path.join(cache_dir, 'pick.pk') 
    with open(out_name, 'rb') as fid:
        all_annots = pickle.load(fid)
    split = int(len(all_annots) * 0.8)
    train_annots, val_annots = all_annots[:split], all_annots[split:]
    convertToVOCFormat(train_annots = train_annots, val_annots = val_annots, output_dir = voc_output_dir, \
                                                    replace_class_name= None, img_shape_dict= None)
    print(f'conver voc dataset job to {voc_output_dir} done')

def main():
    img_path = './data/00000.MTS_100.jpg' # night
    corners = (540, 312, 1069, 819)

    # img_path = './data/0101_1.mov_464.jpg' # day
    # corners = (352, 170, 548, 333)
    img = cv2.imread(img_path)
    img = img[corners[1] : corners[3], corners[0] : corners[2], :]
    # open_img = img_open_filter(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # hist_img = cv2.equalizeHist(gray_img)
    # cv2.imshow('histimg', hist_img)
    ret, gray_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow('openimg', open_img)
    # cv2.imshow('gray_img', gray_img)
    cv2.imwrite('./data/led.jpg', img)
    cv2.imwrite('./data/led_gray.jpg', gray_img)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # test_font_vis()
    gen_score_plate_data()
    gen_voc_dataset()
    # main()