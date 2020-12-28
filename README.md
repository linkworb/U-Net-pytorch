# 基于U-Net的语义分割
# Semantic segmentation based on U-Net
基于[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)修改

> * - [x] Single class
> * - [x] Multi-class


### Dependency
- Pytorch=1.5.1
- Pillow=5.3.0
- Numpy=1.19.1
- Matplotlib=3.2.2
- imgviz=1.2.3

### 1. Train with single class and background
```
python train.py \ 
    --img_dir=./data/imgs
    --mask_dir=./data/masks
    --save_dir=./checkpoints \ 
    --rgb \
    --classes=1 \ 
    --epochs=50 \ 
    --val_percent=0.5 \ 
    ......
```

### 2. Train with multi-class and background
```
python train.py \ 
    ......
    --classes=N \ 
    ......
```

### 3. Train with pretrained model or finetune
```
python train.py \ 
    ......
    --model_dir=./checkppints/model.pth
    ......
```

### 4. predict stage or inference
```
python predict.py \ 
    --model=./checkpoints/49.pth \ 
    --input_dir=./demo \ 
    --viz \ 
    --no-save \ 
    --rgb \ 
    --classes=N
```
* `--viz`: visualize the predict result.
* `--no-save`: Dont not save the predict result.

## Visualization
![Image text](/demo/1.png)
![Image text](/demo/2.png)


