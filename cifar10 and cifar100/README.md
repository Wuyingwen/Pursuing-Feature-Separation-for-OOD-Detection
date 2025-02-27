# Get started
#### Environment
* Python (3.7.13)
* Pytorch (1.13.1)
* torchvision (0.14.1)
* CUDA
* Numpy
#### Pretrained Models and Datasets
Initial models for fine-tuning are provided in folder 
```
./models/
```
Our well-trained models for reproducing the main result are provided in folder
```
./reproduce/
```

Please downloade the following datasets in folder 
```
./data/
```
Auxiliary OOD Dataset
* [Subset of 80 Million Tiny Images](https://github.com/hendrycks/outlier-exposure?tab=readme-ov-file)

Test OOD Datasets
* [SVHN, LSUN, iSUN, Textures, Places365, ImageNet_resize](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

# Training
To train the model on CIFAR benchmarks, simply run:
* CIFAR-10
```
python main.py cifar10 --method ours --score_type msp
```
* CIFAR100
```
python main.py cifar100 --method ours --score_type msp
```
method could be `oe, energy-oe, dal, ours,`, and score_type could be `msp, energy, ours`.

# Test
To test the detection performance of a well-trained model (or reproduce our result in the paper), simply run:
* CIFAR10
```
python test.py cifar10 --score_type ours --model_path ./reproduce/cifar10_ours.pt
```
* CIFAR100
```
python test.py cifar100 --score_type ours --model_path ./reproduce/cifar100_ours.pt
```
where `model_path` is the path to the well-trained model.

# Results
The key results on CIFAR benchmarks are listed in the following table.
 |  |  CIFAR10 | CIFAR10 | CIFAR100 | CIFAR100
 |:-|:---|:---|:---|:---|
 |  | FPR95$\downarrow$ | AUROC$\uparrow$| FPR95$\downarrow$ | AUROC	$\uparrow$
OE | 3.36 | **99.02** | 37.77 | 92.21 
Energy-OE | 2.99 | 98.79 | 42.34 | 91.65 
DAL | 2.69 | 98.86 | 31.47 | 92.82 
Ours | **2.49** | 98.93 | **29.47** | **94.00**
