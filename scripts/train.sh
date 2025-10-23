# pre-train
#export CUDA_VISIBLE_DEVICES="0"
#./tools/dist_train.sh ./projects/SeqGrowGraph/configs/road_seg/lss_roadseg_48x32_b4x8_resnet_adam_24e.py 1

# train
export CUDA_VISIBLE_DEVICES="1,2,3,4"
./tools/dist_train.sh projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_default.py 4





