import json
import os

path      = 'Out_Jsons_Res/ensemble.json'
filejson  = '/data/Dataset/DarkFace/Test_Images/Test_Final.json'
save_path = 'Out_Txts_Res/'
dataset = json.load(open(path, 'r'))
filenames = json.load(open(filejson, 'r'))

id2names = {}
for dict in filenames['images']:
    img_id = dict["id"]
    fname  = dict["file_name"]
    id2names[img_id] = fname

imgid2info = {}
for info in dataset:
    imgid = info['image_id']
    if imgid not in imgid2info:
        imgid2info[imgid] = []
    x1, y1, w, h = info['bbox']
    x2 = x1 + w
    y2 = y1 + h
    score = info['score']
    # if score < 0.1:
    #     continue
    line = '{} {} {} {} {}'.format(x1, y1, x2, y2, score)
    imgid2info[imgid].append(line)

for id in imgid2info:
    filename = id2names[id].replace('.png','.txt')
    info_list = imgid2info[id]
    txt_path = save_path + filename
    with open(txt_path, 'w') as f:
        for line in info_list:
            f.writelines(line + '\n')

print("done")