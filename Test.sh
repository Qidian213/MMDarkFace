#bash tools/dist_test.sh configs/darkface/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_darface.py work_dirs/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_darface/latest.pth 4 --eval bbox

#bash tools/dist_test.sh configs/darkface/cascade_rcnn_r50_fpn_dconv_c3-c5_darface.py work_dirs/cascade_rcnn_r50_fpn_dconv_c3-c5_darface/epoch_80.pth 4 --eval bbox

bash tools/dist_test.sh configs/darkface/cascade_rcnn_r50_fpn_dconv_c3-c5_darface_2.py work_dirs/cascade_rcnn_r50_fpn_dconv_c3-c5_darface_2/epoch_160.pth 4 --eval bbox


python tools/analysis_tools/analyze_results.py configs/darkface/dbhead_r50_fpn_dconv_c3-c5_darface.py db_r50_tta.pkl Out_Imgs

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29513 nohup bash tools/dist_train.sh configs/darkface/cascade_rcnn_r50_fpn_dconv_c3-c5_DCE_WideDark.py 4 >cascade_rcnn_r50_fpn_dconv_c3-c5_DCE_WideDark.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/darkface/dbhead_r50_fpn_dconv_c3-c5_darface.py work_dirs/dbhead_r50_fpn_dconv_c3-c5_darface/epoch_155.pth 4 --eval bbox --out db_r50_tta.pkl
