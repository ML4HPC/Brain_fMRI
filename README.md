# 3D CNN Distributed Deep

## Installation
```{bash}
git clone https://github.com/ML4HPC/Brain_fMRI.git
```

## Training 3D Naive-CNN Model
```{bash}
python3 run_cnn_3d.py --data_dir={Path to data directory} --output_dir={Path to save outputs} --train_batch_size={Train batch size}
--valid_batch_size={Validation batch size} --epoch={# of epochs to train} --optimizer={Optimizer (Adam/SGD)} --normalize={True / False}
```

## Training ResNet50-3D Model
```{bash}
python3 run_resnet3d.py --data_dir={Path to data directory} --output_dir={Path to save outputs} --train_batch_size={Train batch size}
--valid_batch_size={Validation batch size} --epoch={# of epochs to train} --optimizer={Optimizer (Adam/SGD)}
```

Respective model's state after each epoch will be saved to the output directory along with the latest epoch's optimizer's state.
With these saved states, we can resume training for a model with the following command.

```{bash}
python3 run_resnet3d.py --data_dir={Path to data directory} --output_dir={Path to save outputs} --train_batch_size={Train batch size}
--valid_batch_size={Validation batch size} --epoch={# of epochs to train (adjusted from start)} --optimizer={Optimizer (Adam/SGD)} 
--checkpoint_epoch={Epoch # to resume from} --checkpoint_state={Path to model's state file} --checkpoint_opt={Path to optimizer's state file}
