CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop/epoch_24.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./test-run_results"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_anchor_tinaface.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_anchor_tinaface/epoch_24.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_resnest200.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_3000_randomcrop_resnest200/epoch_24.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_4000_randomcrop.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_4000_randomcrop/epoch_36.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_2600_4000_randomcrop_boxshift.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_2600_4000_randomcrop_boxshift/epoch_36.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_gfocal_r2n101_dcn_fpn_ms2x_3000_4000_randomcrop.py work_dirs/face_gfocal_r2n101_dcn_fpn_ms2x_3000_4000_randomcrop/epoch_36.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal/face_strategy_2600_4000.py work_dirs/face_strategy_2600_4000/epoch_36.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/gfocal_new/2600_4000_anchor_refine.py work_dirs/2600_4000_anchor_refine/epoch_36.pth 8 --eval bbox

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/ug2_scale/2000_2800_visdrone_anchor_ori.py work_dirs/2000_2800_visdrone_anchor_ori/epoch_36.pth 8 --format-only --options "jsonfile_prefix=./test-darkface"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 bash tools/dist_test.sh configs/darkface2/2600_4000_anchor_refine_strategy_basescale.py work_dirs/2600_4000_anchor_refine_strategy_basescale/epoch_40.pth 8 --format-only --options "jsonfile_prefix=Out_Jsons_Res/2600_4000_anchor_refine_strategy_basescale_2xmsrcr"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29101 ./tools/dist_test.sh configs/ug2_scale/2000_2800_visdrone_anchor_ori.py work_dirs/2000_2800_visdrone_anchor_ori/epoch_36.pth 8 --format-only --options "jsonfile_prefix=./test-darkface"