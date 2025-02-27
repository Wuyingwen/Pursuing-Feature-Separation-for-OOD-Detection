# Get started

#### Environment

* Python (3.7.13)
* Pytorch (1.13.1)
* torchvision (0.14.1)
* CUDA
* Numpy

#### Pretrained Datasets

Please downloade the following datasets in folder

```
./data/
```

Auxiliary OOD Dataset

```
wget -c https://www.image-net.org/data/imagenet21k_resized.tar.gz
```

Test OOD Datasets: refer to [KNN](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

# Train

```
bash imagenet_1k_finetune_oe.sh
bash imagenet_1k_finetune_dal.sh
bash imagenet_1k_finetune_ours.sh
```
