# QOT
QUATERNION ORTHOGONAL TRANSFORMER FOR FACIAL EXPRESSION RECOGNITION IN THE WILD
## Requirements
- Python=3.8
- tensorflow=2.6.0
- PyTorch=1.10
- torchvision=0.11.0
- cudatoolkit=11.3
- matplotlib=3.5.3

## Training & Evaluate
We evaluate QOT on RAF-DB, AffectNet and SFEW. We take RAF-DB as an example to show our methods.
- Step 1: download RAF-DB datasets from offical website, and put it into ./datasets
- Step 2: download pre-trained ResNet-50 from Google Drive, and put it into ./pretrianed
- Step 3: run main_Upload.py to train Orthogonal_CNN model.
- Step 4: replace the path with the pretrained model in Step 3 in main_generate_ortho.py to generate the numpy file of orthogonal features.
- Step 5: load orthogonal features generated in Step4 or directly download from Google Drive, and run q-vit_RAFDB_Upload.py.

## Pre-trained Model


