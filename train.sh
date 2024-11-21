# python train.py -s ~/data/zju_mocap_refine/my_386 --eval --exp_name zju_mocap_refine/my_386_upsampled --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 5000 
# python train.py -s ~/data/zju_mocap_refine/my_392 --eval --exp_name zju_mocap_refine/my_392_env --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 5000 
# python train.py -s ~/data/zju_mocap_refine/my_377 --eval --exp_name zju_mocap_refine/my_377_upsampled --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 5000
# python train.py -s ~/data/render/eric/eric_train --eval --exp_name render/eric --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 5000
python train.py -s ~/data/mixamo/ch21 --eval --exp_name mixamo/ch21_entropy --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 7000 
# python train.py -s ~/data/mixamo/ch26 --eval --exp_name mixamo/ch26 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 7000 --start_checkpoint output/mixamo/ch21/chkpnt5000.pth

# python train.py -s ~/data/mixamo/ch37 --eval --exp_name mixamo/ch37 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 7000 --start_checkpoint output/mixamo/ch37/chkpnt5000.pth

# python train.py -s ~/data/mixamo/ch38 --eval --exp_name mixamo/ch38 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 5000 