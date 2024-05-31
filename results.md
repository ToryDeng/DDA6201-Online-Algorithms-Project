# data info
8487 cells in the test dataset


# vanilla transformer
test f1: 0.7037793397903442
test acc: 0.7800164818763733
test latency: 00:45

# gated transformer + supervised pretraining
- saved model: epoch=47_val_f1=0.6034
test f1: 0.6439070701599121
test acc: 0.7116766571998596
test latency: 00:59

# gated transformer + supervised pretraining + rl
test f1: 0.7204700708389282
test acc: 0.7809591293334961
test latency: 00:37


python train_rl.py --saved_dir ckpt/reinforce/version_25/epoch=0_best_score=0.6945.pth