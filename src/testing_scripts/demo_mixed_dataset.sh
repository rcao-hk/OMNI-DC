# model configs
load_dav2=1
resolution=3
pred_confidence_input=1
multi_resolution_learnable_gradients_weights="uniform"
optim_layer_input_clamp=1.0
depth_activation_format='exp'
max_depth=5.0
whiten_sparse_depths=1
backbone='rgbd'

# checkpoints
ckpt=../checkpoints/modelv1.1_best_72epochs.pt
# ckpt=../checkpoints/model_best_72epochs.pt

# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 4 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'HAMMER' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_518x518'  --camera 'tof'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'HouseCat6D' --split '/home/robotarm/object_depth_percetion/dataset/splits/HouseCat6D_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'TransCG' --split '/home/robotarm/object_depth_percetion/dataset/splits/TransCG_d435_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518' --camera 'd435'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'GN-Trans' --split '/home/robotarm/object_depth_percetion/dataset/splits/GN-Trans_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'XYZ-IBD' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'YCB-V' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518'


# python infer_mixed_dataset.py \
#    --max_depth $max_depth --data_normalize_median 1 \
#    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
#    --load_dav2 $load_dav2 \
#    --gpus 2 \
#    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#    --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
#    --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
#    --pretrain $ckpt --dataset 'T-LESS' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_v0_518x518'

python infer_mixed_dataset.py \
   --max_depth $max_depth --data_normalize_median 1 \
   --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
   --load_dav2 $load_dav2 \
   --gpus 5 \
   --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
   --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method mean \
   --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
   --pretrain $ckpt --dataset 'ROBI' --split '/home/robotarm/object_depth_percetion/dataset/splits/ROBI_test.txt' --dataset_root '/data/robotarm/dataset' --img_height 480 --img_width 640 --output_root '/data/robotarm/result/depth/mixed' --method 'omnidc_zs_480x640'