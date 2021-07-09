from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
import os
import json 
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.01, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, out_file='res.jpg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.01, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    data_dir ='/data/Dataset/DarkFace/Test_Images/Track1.2_testing_samples_MSRCR/' #'/data/Dataset/DarkFace/Test_Images/Test_Dce_Enlight/'
    files = os.listdir(data_dir)
    result_dt = {}
    for file in files:
        print(file)
        imgname = data_dir + file
        result = inference_detector(model, imgname)
        
      #  show_result_pyplot(model, imgname, result, score_thr=args.score_thr, out_file='Out_Imgs/' + file)
        
        bboxes = np.vstack(result)

        str_lists = []
        for box in bboxes:
            line = str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4])
            str_lists.append(line) 
                
        with open('Out_Txts/' + file.replace('png', 'txt'),'w') as f:
            f.write('\n'.join(str_lists))

        boxs = []
        for box in bboxes:
            boxs.append([float(box[0]),float(box[1]),float(box[2]),float(box[3]),float(box[4])])
        result_dt[file] = boxs

    json.dump(result_dt, open('Out_Jsons/MS1.json', 'w'), indent=4)
    
#CUDA_VISIBLE_DEVICES=0 python GenRes.py configs/darkface/cascade_rcnn_r50_fpn_dconv_c3-c5_darface.py work_dirs/cascade_rcnn_r50_fpn_dconv_c3-c5_darface/epoch_80.pth