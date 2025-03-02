# UTeRM
Source code and data for the paper  
Deep Unfolding Tensor Rank Minimization With Generalized Detail Injection for Pansharpening  
Truong Thanh Nhat Mai, Edmund Y. Lam, and Chul Lee  
IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-18, 2024, Art no. 5405218  
https://doi.org/10.1109/TGRS.2024.3392215

For PDF, please visit https://mtntruong.github.io/

If you have any questions, please open an issue.  

# Source code
The proposed algorithm is implemented in Python using PyTorch. Since the networks do not require any special Python libraries other than Pytorch, you can easily plug them into your training code to train with your own dataset. Please see the toy example in the `main` function in each file.

Please note that the proposed UTeRM needs data augmentation to achieve the reported performance (we also used the same data augmentation procedure for all deep networks in the experiments to ensure fair comparisons).

## Required Python packages
Even though the deep unfolded networks only need Pytorch to run, the training/testing scripts require some external libraries. Please use `env.yml` to create an environment with [Anaconda](https://www.anaconda.com)
```
conda env create -f env.yml
```
Then activate the environment
```
conda activate uterm
```
If you want to change the environment name, edit the first line of `env.yml` before creating the environment.

## Data Preparation
Please download the raw data from [here](https://github.com/Lihui-Chen/Awesome-Pansharpening#Datasets) then run the MATLAB script `Data-Preparation/H5_Data_Gen.m` to create an H5 file containing preprocessed and augmented data. Please note that all learning-based competing algorithms in the paper use this offline-augmented data for training.

## Training
The following script performs the training procedure described in the paper
```
python train.py --arch UTeRM_CNN --data ./msi_data/IKONOS_train.h5 && python train.py --arch UTeRM_CNN --data ./msi_data/IKONOS_train.h5 --finetune --resume=./ckpts/epoch_90.pth --set_lr=1e-6
```
The data file `./msi_data/IKONOS_train.h5` serves as an example of the generated H5 file described above.

## Testing
Reduced-resolution test
```
python test_reduced_res.py --arch UTeRM_CNN --data ./msi_data/IKONOS_test.h5 --weight ./checkpoints/UTeRM_CNN.pth
```
Full-resolution test
```
python test_full_res.py --arch UTeRM_CNN --data ./msi_data/IKONOS_test.h5 --weight ./checkpoints/UTeRM_CNN.pth
```
Please note that the file `IKONOS_test.h5` only contains a part of the test set used in the paper since GitHub does not allow big files.

# Citation
If our research or dataset are useful for your research, please kindly cite our work
```
@article{Mai2024,
    author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Deep Unfolding Tensor Rank Minimization With Generalized Detail Injection for Pansharpening}, 
    year={2024},
    volume={62},
    number={},
    pages={1-18},
    doi={10.1109/TGRS.2024.3392215}
}
```
