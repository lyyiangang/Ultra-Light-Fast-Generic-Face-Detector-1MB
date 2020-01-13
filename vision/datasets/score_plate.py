import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import ipdb
from vision.utils import crash


class ScorePlate:
    def __init__(self):
        # m m : s s : ms
        # circle ss . ms
        # original image: plate size(500, 500), the 2 lines text take whole image
        self.img = np.zeros((200,600,3),np.uint8)
        fontpath = "data/pixel_lcd7/Pixel LCD-7.ttf"
        print(f'loading font from {fontpath}')
        self.font = ImageFont.truetype(fontpath, 80)
   
    def render_time(self, m_up, s_up, ms_up):
        """render time on image according input time
        Returns:
            (image, annot_list)
            image: np.array
            annot_list: [(cls_name, np.array((x1, y1, x2, y2)), (cls_name, np.array(x1, y1, x2, y2))]
        """
        assert m_up < 99 and m_up >= 0 and s_up <= 60 and s_up >= 0 
        self.img[:, :, :] = 0
        img_pil = Image.fromarray(self.img)
        candidate_colors = [(0, 0, 255, 0), (0, 255, 0, 0), (255, 0, 0, 0)]
        color_idx = np.random.choice(len(candidate_colors))
        draw = ImageDraw.Draw(img_pil)
        up_line = f'{m_up}:{s_up}.{ms_up}'
        start_pt = (50, 50)
        draw.text(start_pt ,  up_line, font = self.font, fill = candidate_colors[color_idx])
        rect_x = start_pt[0]
        annot_list = []
        for cc in up_line:
            offset = self.font.getoffset(cc)
            ww, hh = self.font.getmask(cc).size
            tl = (rect_x + offset[0], start_pt[1] + offset[1]) 
            rb = (tl[0] + ww, tl[1] + hh) 
            raw_bbox = tl + rb
            corrected_bbox = self.correct_bbox(cc, raw_bbox)
            #('1', 'detection', (5, 6, 7, 8))
            annot_list.append((cc, 'detection', tuple(corrected_bbox)))
            # draw.rectangle(corrected_bbox)
            rect_x += ww
        img = np.array(img_pil)
        return img, annot_list

    def correct_bbox(self, char, raw_bbox):
        # ipdb.set_trace()
        raw_bbox = np.array(raw_bbox)
        w, h = raw_bbox[2] - raw_bbox[0], raw_bbox[3] - raw_bbox[1]
        if char == ':':
            new_w, new_h = w / 6, h / 1.8
            left_up = np.array((raw_bbox[0] + w / 2.5, raw_bbox[1]))
        elif char == '.':
            new_w, new_h = w / 6, h / 2
            left_up = np.array((raw_bbox[0] + w / 2.5, raw_bbox[1]))
        elif char == '1':
            new_w, new_h = w / 5, h
            left_up = np.array((raw_bbox[0] + w /1.4, raw_bbox[1]))
        else:
            new_w, new_h = w / 1.1, h
            left_up = np.array(raw_bbox[:2])
        right_bottom = left_up + np.array([new_w, new_h])
        raw_bbox = np.array([left_up, right_bottom]).flatten()

        return raw_bbox

    @staticmethod
    def render_annots(image, annot_list):
        for annot in annot_list:
            cc, _, bbox = annot
            bbox = np.array(bbox).reshape(2, 2).astype(np.int32)
            cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255)) 


def test_score_palte():
    plate = ScorePlate()
    img, annots = plate.render_time(1, 30, 50)
    print(f'annots:{annots}')
    ScorePlate.render_annots(img, annots)
    cv2.imshow('result', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    test_score_palte()
