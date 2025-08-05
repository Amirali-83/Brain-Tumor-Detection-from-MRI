# 🧠 Brain-Tumor-Detection-from-MRI

### Project Overview
This project implements a U-Net architecture with a ResNet34 encoder to perform semantic segmentation of brain tumors in MRI scans. The model is trained on 2D slices extracted from the BraTS 2020 dataset (converted to .h5 format for faster loading) and achieves pixel-wise tumor detection with interpretable visualisations.

------------
### Model Architecture
Backbone: ResNet34 (ImageNet pretrained)
Segmentation head: U-Net
Loss Function: Combined Dice Loss + CrossEntropyLoss
Framework: PyTorch + segmentation-models-pytorch
Precision: Automatic Mixed Precision (AMP) for faster training on T4 GPUs

----------
### Performance
Dice Coefficient: ~0.80 (on validation set)
IoU Score: ~0.70
Visual output includes MRI, true mask, predicted mask, and overlay
Includes automated radiology-style interpretation reports
