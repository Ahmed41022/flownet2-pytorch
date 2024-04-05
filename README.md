# flownet2-pytorch

Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925).

Multiple GPU training is supported, and the code provides examples for training or inference on [MPI-Sintel](http://sintel.is.tue.mpg.de/) clean and final datasets. The same commands can be used for training or inference with other datasets. See below for more detail.

Inference using fp16 (half-precision) is also supported.

For more help, type <br />

    python main.py --help

## Network architectures

- **FlowNet2**

## Data Loaders

Dataloaders for FlyingChairs, FlyingThings, ChairsSDHom and ImagesFromFolder are available in [datasets.py](./datasets.py). <br />

## Loss Functions

L1 and L2 losses with multi-scale support are available in [losses.py](./losses.py). <br />

### Python requirements

- numpy
- PyTorch
- scipy
- scikit-image
- tensorboardX
- colorama, tqdm, setproctitle

## Converted Caffe Pre-trained Models

We've included caffe pre-trained models. Should you use these pre-trained weights, please adhere to the [license agreements](https://drive.google.com/file/d/1TVv0BnNFh3rpHZvD-easMb9jYrPE2Eqd/view?usp=sharing).

- [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]

## Inference

    # Example on MPISintel Clean
    python main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean \
    --inference_dataset_root /path/to/mpi-sintel/clean/dataset \
    --resume /path/to/checkpoints

## Training and validation

    # Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
    python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
    --training_dataset MpiSintelFinal --training_dataset_root /path/to/mpi-sintel/final/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset

    # Example on MPISintel Final and Clean, with MultiScale loss on FlowNet2C model
    python main.py --batch_size 8 --model FlowNet2C --optimizer=Adam --optimizer_lr=1e-4 --loss=MultiScale --loss_norm=L1 \
    --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --crop_size 384 512 \
    --training_dataset FlyingChairs --training_dataset_root /path/to/flying-chairs/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset

## Reference

If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper:

```
@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
```

```
@misc{flownet2-pytorch,
  author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
  title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}
```

## Related Optical Flow Work from Nvidia

Code (in Caffe and Pytorch): [PWC-Net](https://github.com/NVlabs/PWC-Net) <br />
Paper : [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371).

## Acknowledgments

Parts of this code were derived, as noted in the code, from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
