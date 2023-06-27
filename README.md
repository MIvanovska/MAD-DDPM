# MAD-DDPM
Official implementation of the paper "Face Morphing Attack Detection with Denoising Diffusion Probabilistic Models" [^1]. 

MAD-DDPM is a one-class learning model that uses a reconstruction-based measure to determine whether the input images are bona fide or face morphs. At the core of the technique is a two-branch reconstruction procedure that uses denoising diffusion probabilistic models (DDPMs) learned over only bona-fide samples as the basis for the
detection tasks. The first branch models the distribution on bona-fide samples directly in the pixel-space (for low-level artifact detection), while the second captures the distribution of higher-level features extracted with a pretrained CNN.
      

![MAD-DDPM](MAD_DDPM.png)

For more information please refer to the paper available [here](https://lmi.fe.uni-lj.si/wp-content/uploads/2023/06/IWBF2023___Face_Morphing_Attack_Detection_with_Denoising_Diffusion_Probabilistic_Models.pdf).

## 1. Install the dependencies
The model is implemented using PyTorch. The full list of used libraries can be found in requirements.txt.
```
pip install -r requirements.txt
```

## 2. Prepare the data
The datasets you are using for training or testing should be placed in the directory called data. Datasets should have the following directory structure:
```
Dataset_name
├── images
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.png
│   │   │   └── bonafide_img_2.png
│   │   │   └── ...
│   ├── morphs
│   │   ├── morphing_method_1
│   │   │   └── morph_img_1.png
│   │   │   └── morph_img_2.png
│   │   │   └── ...
│   │   ├── morphing_method_2
│   │   ...
├── features_scale_1
│   ├── bonafide
│   │   ├── raw
│   │   │   └── bonafide_img_1.pt
│   │   │   └── bonafide_img_2.pt
│   │   │   └── ...
│   ├── morphs
│   │   ├── morphing_method_1
│   │   │   └── morph_img_1.pt
│   │   │   └── morph_img_2.pt
│   │   │   └── ...
│   │   ├── morphing_method_2
│   │   ...
├── features_scale_2
│   ...
```

Images are expected to have one of the following image extensions: '.jpg', '.jpeg' or '.png'. Their corresponding pre-extracted feature maps should be saved with the same name in '.pt' format (PyTorch tensors). features_scale_1 is the root directory of features extracted from cropped face images of size 224x224 pixels (tensor size is 1024x14x14), while features_scale_2 contains features of images of size 448x448 pixels (tensor size is 4x1024x14x14).

MAD-DDPM is trained and tested on preprocessed datasets, where faces were first detected with RetinaFace, then cropped out with a margin of 5% of the detected bounding box height. Corresponding feature maps are extracted with a pretrained WideResNet. For more details please refer to the paper.

## 3. Training
To train the **image branch** MAD-DDPM on your dataset, run the following:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 1
```
To train with evaluation on a testing set after each training epoch run:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 1 --test-every 1 --test-set ./data/FRGC/
```
To train the **branch for features**, set the value of the argument branch to 2:
```
python train.py --train-set ./data/CASIA-WebFace/ --config configs/casia_webface.json --branch 2
```
Checkpoints are by default exported to the directory named output. 

## 4. Evaluation
To test a pretrained MAD-DDPM model run the following:
```
python test.py --test-set ./data/FRGC/ --image_branch_checkpoint output/experiment/checkpoints/model_image.pth --features_branch_checkpoint output/experiment/checkpoints/model_features.pth --config configs/casia_webface.json
```
## 5. Citing MAD-DDPM
If you find this code useful or you want to refer to the paper, please cite using the following BibTeX:
```
@INPROCEEDINGS{ivanovska2023mad_ddpm,
  author={Ivanovska, Marija and Štruc, Vitomir},
  booktitle={2023 11th International Workshop on Biometrics and Forensics (IWBF)}, 
  title={Face Morphing Attack Detection with Denoising Diffusion Probabilistic Models}, 
  year={2023},
  doi={10.1109/IWBF57495.2023.10156877}}}
```

## Acknowledgements
This code is largely based on [k-diffusion](https://github.com/crowsonkb/k-diffusion).

  
## References
[^1]: M. Ivanovska, V. Struc, Face Morphing Attack Detection with Denoising Diffusion Probabilistic Models, International Workshop on Biometrics and Forensics (IWBF), 2023
