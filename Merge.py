import json
import cv2
import numpy as np
from mmcv.ops.nms import soft_nms

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# fp1   = open('Out_Jsons/MS1.json','r') 
# data1 = json.load(fp1)

# fp2   = open('Out_Jsons/MS2.json','r') 
# data2 = json.load(fp2)

# data_dir = '/data/Dataset/DarkFace/Test_Images/Test_Dce_Enlight/'
    
# for key in data1.keys():
    # anns1 = data1[key]
    # anns2 = data2[key]

    # anns = anns1 + anns2
    # anns = np.array(anns).astype(np.float32)
    
    # dets   = anns[:, :4]
    # scores = anns[:, 4:]
    
    # bboxs,_ = soft_nms(dets, scores, 0.4)
    
    # print(key, len(anns1), len(anns2), len(anns) , len(bboxs))
    
    # str_lists = []
    # for box in bboxs:
        # line = str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4])
        # str_lists.append(line) 
        
    # with open('Out_Txts/' + key.replace('png', 'txt'),'w') as f:
        # f.write('\n'.join(str_lists))
        
    # imgname = data_dir + key
    # image   = cv2.imread(imgname)
    
    # for box in bboxs:
        # cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [00,255,255], 2, 16)
    # cv2.imwrite('Out_Imgs/' + key, image)

fp1   = open('Out_Jsons/MS1.json','r') 
data1 = json.load(fp1)

# fp2   = open('Out_Jsons/MS2.json','r') 
# data2 = json.load(fp2)

# fp3   = open('Out_Jsons/MS3.json','r') 
# data3 = json.load(fp3)

# fp4   = open('Out_Jsons/MS4.json','r') 
# data4 = json.load(fp4)

data_dir = '/data/Dataset/DarkFace/Test_Images/Test_Dce_Enlight/'
    
for key in data1.keys():
    anns1 = data1[key]
  # anns2 = data2[key]
   # anns3 = data3[key]
    # anns4 = data4[key]
    
    anns = anns1# + anns2
    anns = np.array(anns)
    
    keeps = nms(anns, 0.5)
    bboxs = anns[keeps]
    
    print(key, len(anns1), len(anns2), len(anns) , len(bboxs))
    
    str_lists = []
    for box in bboxs:
        line = str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4])
        str_lists.append(line) 
        
    with open('Out_Txts/' + key.replace('png', 'txt'),'w') as f:
        f.write('\n'.join(str_lists))
        
    imgname = data_dir + key
    image   = cv2.imread(imgname)
    
    for box in bboxs:
        cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1), [00,255,255], 2, 16)
    cv2.imwrite('Out_Imgs/' + key, image)
    