# RACNN-pytorch
This is a third party implementation of RA-CNN in pytorch. I am still working on reproducing a same performance written in [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/Look-Closer-to-See-Better-Recurrent-Attention-Convolutional-Neural-Network-for-Fine-grained-Image-Recognition.pdf)

## Requirements
- python3
- [Pytorch 0.4 > ](https://github.com/pytorch/pytorch#from-source)
- torchvision
- numpy
- [tensorflow](https://www.tensorflow.org/install/), optional

## TODO
- [x] Network building
- [ ] Repactoring for arguments
- [ ] Pre-training a APN
- [ ] Alternative training between APN and ConvNet/Classifier
- [ ] Reproduce and report on README.md
- [ ] Sample visualization
- [ ] Add new approach to improve

## Current issue
- Don't know how to pre-train a APN. Need more details
- Rankloss doesn't decrease. Because no pretrain? or bugs?

## Results

Current best is 71.68% at scale1 without APN pretraining. It's bad than using just VGG19

## Usage

For training, use following command.

```bash
$ python trainer.py
```

Currently only cuda available device support.

If you want to see training process,

```bash
$ Tensorboard --log='visual/' --port=6666
```

and go to 'localhost:6666' on webbrowser. You can see the Loss, Acc and so on.

## References

- [Original code](https://github.com/Jianlong-Fu/Recurrent-Attention-CNN)
- [Other pytorch implementation](https://github.com/Charleo85/DeepCar)
    - with car dataset, I refer the attention crop code from here
