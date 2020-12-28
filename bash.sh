python train.py --save_dir=./checkpoints --rgb --classes=3 --epochs=50 --val_percent=0.5
python predict.py --model=./checkpoints/49.pth --input_dir=./demo --viz --no-save --rgb --classes=3