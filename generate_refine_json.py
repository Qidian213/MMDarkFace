# -*- coding:utf-8 -*-
# !/usr/bin/env python
import os
import xml.dom.minidom as xmldom
import numpy as np
import glob
from PIL import Image
import json
import random
import cv2

class xml2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./new.json'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = [
            {'id': 1, 'name': 'face'}]

        # 'person', 'car', 'bus', 'bicycle', 'motorbike'
        self.annotations = []
        self.label = []
        self.annID = 1
        self.imgID = 1
        self.height = 0
        self.width = 0
        self.save_json()

    def data_transfer(self):
        for num, xmlfile in enumerate(self.labelme_json):
            DomTree = xmldom.parse(xmlfile)
            annotation = DomTree.documentElement


            objectlist = annotation.getElementsByTagName('object')
            write_box = False
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                label = (namelist[0].childNodes[0].data)  # .encode('unicode-escape').decode('string_escape')
                bndbox = objects.getElementsByTagName('bndbox')
                if (len(bndbox) > 0):
                    # self.annotations.append(self.annotation(bndbox, label, num))
                    tmp_ann = self.annotation(bndbox, label, num)
                    if tmp_ann != {}:
                        self.annotations.append(tmp_ann)
                        self.annID += 1
                        write_box = True
            if write_box:
                self.images.append(self.image(annotation, num))



    def image(self, annotation, num):
        image = {}
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        image['file_name'] = filename
        size_node = annotation.getElementsByTagName("size")
        for size_ in size_node:
            width_node = size_.getElementsByTagName("width")[0]
            width = int(width_node.childNodes[0].data)
            height_node = size_.getElementsByTagName("height")[0]
            height = int(height_node.childNodes[0].data)
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1

        self.height = height
        self.width = width

        return image

    def annotation(self, bndbox, label, num):
        annotation = {}
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(float(x1_list[0].childNodes[0].data))
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(float(y1_list[0].childNodes[0].data))
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(float(x2_list[0].childNodes[0].data))
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(float(y2_list[0].childNodes[0].data))

            assert  x1 <= x2 and y1 <= y2
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                print("w = ", w, "  h = ",h)

            annotation['bbox'] = [x1, y1, w, h]
        if 'bbox' not in annotation:
            return {}

        annotation['segmentation'] = None
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID

        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        print("Num categories: %s" % len(self.categories))
        print("Num images: %s" % len(self.images))
        print("Num annotations: %s" % len(self.annotations))

        # img_path = '/data/Dataset/UG2/Reside-B/Unannotated/images/'
        # img_filenames = os.listdir(img_path)
        # img_ann = [x['file_name'] for x in data_coco['images']]
        # img_noann = [x for x in img_filenames if x not in img_ann]
        # start_id = data_coco['images'][-1]['id'] + 1
        # tmp_images = []
        # for name in img_noann:
        #     file_name = name
        #     id = start_id
        #     start_id += 1
        #     img = cv2.imread(img_path + name)
        #     height, width = img.shape[0], img.shape[1]
        #     tmp = {
        #         'file_name':file_name,
        #         'height':height,
        #         'width':width,
        #         'id':id
        #     }
        #     tmp_images.append(tmp)
        # data_coco['images'].extend(tmp_images)
        # print("--------------- ADD noann images---------------")
        # print("Num categories: %s" % len(data_coco['categories']))
        # print("Num images: %s" % len(data_coco['images']))
        # print("Num annotations: %s" % len(data_coco['annotations']))


        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


xml_files = glob.glob('/data/Dataset/UG2/darkface/coco/refine/471_20210422195402/dark_face_annotated_20210422195402/*.xml', recursive=True)
random.shuffle(xml_files)

train_xml_files = xml_files

save_json_train = '/data/Dataset/UG2/darkface/coco/refine/471_20210422195402/refine.json'



xml2coco(train_xml_files, save_json_train)

