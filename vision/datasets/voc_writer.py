import os
import numpy as np
import cv2
import time
import glob
import json
import shutil
import xml.etree.cElementTree as ET
from xml.dom import minidom
import logging
lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

def convertToVOCFormat(train_annots, val_annots, output_dir : str, replace_class_name = None, img_shape_dict = None):
    """
    convert annotion files to VOC format. the directories structure:
    output_dir
        └── VOC
            ├── Annotations       
            ├── ImageSets         
            │   ├── Action        
            │   ├── Layout        
            │   ├── Main (train.txt, trainval.txt, val.txt)
            │   └── Segmentation  
            ├── JPEGImages        
            ├── SegmentationClass 
            └── SegmentationObject
    for detection annotation files, only "Annotations", "ImageSets" and "JPEGImages" will be generated.
    replace_class_name: dict. replace old class name to a new class name
    train_annots and val_annots should be in format: 
    [
        (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
        (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
        (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
        ...
    ]
    img_shape_dict: dictionary, which holds the height and width of input images. if the param is not None,
    the 'cv2.imread' will not used in this function, and it will speed up the whole routine. 
    the img_shape_dict in format{'img_path_a' : [h, w], 'img_path_b' :[h, w], ...}, how to use it, pls see 'test_convert_voc_format'
    """
    voc_root_dir = 'VOC'
    annotation_dir = os.path.join(output_dir, voc_root_dir, 'Annotations')
    os.makedirs(annotation_dir, exist_ok= True)
    imgset_dir = os.path.join(output_dir, voc_root_dir, 'ImageSets')
    imgset_main_dir = os.path.join(imgset_dir, 'Main')
    os.makedirs(imgset_main_dir, exist_ok= True)
    jpegs_dir = os.path.join(output_dir, voc_root_dir, 'JPEGImages')
    os.makedirs(jpegs_dir, exist_ok= True)

    train_imgs = [item[0] for item in train_annots]
    val_imgs = [item[0] for item in val_annots]
    imgpath_with_annots = train_annots + val_annots
    if img_shape_dict is not None:
        assert len(img_shape_dict) == len(imgpath_with_annots), 'the length should be euqal'
    idx = 0
    fid_train = open(os.path.join(imgset_main_dir, 'train.txt'), 'w')
    fid_val = open(os.path.join(imgset_main_dir, 'val.txt'), 'w')
    for img_path, annots in imgpath_with_annots:
        if img_path is None:
            continue
        if idx % 1000 == 0:
            lg.info('process {}/{}'.format(idx, len(imgpath_with_annots)))
        idx += 1
        if img_shape_dict:
            hh, ww = img_shape_dict[img_path]
            img = None
        else:
            img = cv2.imread(img_path)
            hh, ww = img.shape[:2]
        jpg_name = os.path.splitext(os.path.basename(img_path))[0] + '_{}.jpg'.format(idx)
        if not os.path.splitext(img_path)[-1] == '.jpg':
            lg.warn('ready to convert img {} to jpeg format'.format(img_path))
            if img is None:
                img = cv2.imread(img_path)
            ret = cv2.imwrite(os.path.join(jpegs_dir, jpg_name), img)
            assert ret, 'error occur when writing {}'.format(jpg_name)
        else:
            shutil.copy(img_path, os.path.join(jpegs_dir, jpg_name))

        base_name = os.path.splitext(jpg_name)[0] 
        annot_xml = os.path.join(annotation_dir, base_name + '.xml')
        _dump_annotations_to_file(annot_xml, jpg_name, voc_root_dir, annots, (ww, hh), replace_class_name)
        if img_path in train_imgs:
            fid_train.write('{}\n'.format(base_name))
        elif img_path in val_imgs:
            fid_val.write('{}\n'.format(base_name))
        else:
            assert False
    lg.info('writing image list to file. {} train images and {} val images, the voc dataset is saved to {}'.format(len(train_imgs), len(val_imgs), output_dir))
    fid_train.close()
    fid_val.close()
            
def _dump_annotations_to_file(xml_file, img_file_name, folder, annots, img_size, replace_class_name : dict):
    """
    write detection annotations to xml_file.
    annotation format:
    <annotation>
        <filename>2011_002810.jpg</filename>
        <folder>VOC2012</folder>
        <size>
            <width>486</width>
            <height>500</height>
            <depth>3</depth>
        </size>
        <object>
            <name>tvmonitor</name>
            <bndbox>
                <xmax>91</xmax>
                <xmin>28</xmin>
                <ymax>259</ymax>
                <ymin>201</ymin>
            </bndbox>
            <difficult>0</difficult>
            <occluded>1</occluded>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
        </object>
        <object>
        #....obj2
        </object> 
    </annotation>
    """
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = img_file_name
    ET.SubElement(annotation, 'folder').text = folder
    size_node = ET.SubElement(annotation, 'size')
    ET.SubElement(size_node, 'width').text = str(img_size[0])
    ET.SubElement(size_node, 'height').text = str(img_size[1])
    exist_detection_annotations = False
    for anno in annots:
        cls, annot_type, pts_or_rect = anno
        if replace_class_name is not None:
            cls = replace_class_name[cls]
        if annot_type == 'segmentation':
            pts_or_rect = np.array(pts_or_rect).astype(np.int32).reshape(-1, 2)
            (xmin, ymin), (xmax, ymax) = np.min(pts_or_rect, axis = 0), np.max(pts_or_rect, axis = 0)
        elif annot_type == 'detection':
            xmin, ymin, xmax, ymax = int(pts_or_rect[0]), int(pts_or_rect[1]), int(pts_or_rect[2]), int(pts_or_rect[3])
        else:
            assert False, 'unknown annotation type'
        exist_detection_annotations = True
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = cls
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    if exist_detection_annotations:
        xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
        with open(xml_file, "w") as f:
            f.write(xmlstr)



#  [
#         (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
#         (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
#         (img_path, [(cls, annot_type, (p1x, p1y, p2x, p2y)), (cls, annot_type, (p2x, p1y, p2x, p2y))])
#         ...
#     ]
#     # lg.info(img_shape_dict)
def test():
    train_annots = [
                        ('data/00000.MTS_100.jpg', [('0', 'detection', (1, 2, 3, 4)),
                                   ('1', 'detection', (5, 6, 7, 8))
                               ]
                        )
                    ]
    val_annots = [
                        ('data/00000.MTS_100.jpg', [('0', 'detection', (1, 2, 3, 4)),
                                   ('1', 'detection', (5, 6, 7, 8))
                               ]
                        )
                    ]
    
    convertToVOCFormat(train_annots = train_annots, val_annots = val_annots, output_dir = '../../Dataset', \
                                                    replace_class_name= None, img_shape_dict= None)

if __name__ == '__main__':
    test()
    print('done')