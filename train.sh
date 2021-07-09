CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29201 ./tools/dist_train.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_tinaface.py 4


CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29201 ./tools/dist_train.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_tinaface_scale.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29201 ./tools/dist_train.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_anchor_tinaface.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29201 ./tools/dist_train.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_resnest200.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29201 nohup ./tools/dist_train.sh configs/gfocal_new/2600_4000_anchor_refine_strategy.py 8 >2600_4000_anchor_refine_strategy.txt 2>&1 &