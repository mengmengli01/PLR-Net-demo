# PLR-Net_torch_version
Official Pytorch Code base for "Extracting vectorized agricultural parcels from high-resolution satellite images using a Point-Line-Region interactive multi-task model"
<div align="center">
  <img src="PLRNet/Fig_2_Methods.png">
</div>

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
A GF2 image (1m) and training weight is provided for scientific use: https://pan.baidu.com/s/16h4mlkxFfaOuX1HRDPholQ, password：hql4
A GF2 image (1m) is provided for scientific study: https://drive.google.com/file/d/1JZtRSxX5PaT3JCzvCLq2Jrt0CBXqZj7c/view?usp=drive_link A corresponding partial field label is provided: https://drive.google.com/file/d/19OrVPkb0MkoaUvaax_9uvnJgSr_dcSSW/view?usp=sharing
# Sentinel-2 dataset
The remote sensing image of the Netherlands study area was obtained from PDOK (https://collections.eurodatacube.com/), and the ground truth for the Sentinel-2 dataset in Netherlands were obtained from https://www.pdok.nl/.
