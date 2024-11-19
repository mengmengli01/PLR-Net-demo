# PLR-Net_torch_version
Official Pytorch Code base for "Extracting vectorized agricultural parcels from high-resolution satellite images using a Point-Line-Region interactive multi-task model"
# Training & Testing

  The structure of the data file should be like:
  
  /data 
  
  |-- data
  
      |-- train
      |   |-- images
      |   |-- annotation.json

      |-- val
      |   |-- images
      |   |-- annotation.json

      
  Training
  
  Single GPU training
  
  python scripts/train.py --config-file config-files/PLR-Net.yaml 
  
  Testing
  
  python scripts/test.py --config-file config-files/PLR-Net.yaml 

# GF-2 dataset
A GF2 image (1m) is provided for scientific use: https://pan.baidu.com/s/1isg9jD9AlE9EeTqa3Fqrrg, password：bzfd
