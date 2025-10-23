# test for pre-train
# CUDA_VISIBLE_DEVICES="1" python3 ./tools/test.py \
#    projects/SeqGrowGraph/configs/road_seg/lss_roadseg_48x32_b4x8_resnet_adam_24e.py \
#    ckpts/lss_roadseg_48x32_b4x8_resnet_adam_24e/epoch_24.pth


# test 
# single-gpu
# CUDA_VISIBLE_DEVICES="1" python3  ./tools/test.py \
#     projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_default.py \
#     work_dirs/seq_grow_graph/last_checkpoint 


# multi-gpu   
./tools/dist_test.sh ./tools/dist_test.sh projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_default.py \
     work_dirs/seq_grow_graph/last_checkpoint 8
