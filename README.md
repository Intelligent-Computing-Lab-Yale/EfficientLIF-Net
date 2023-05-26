# EfficientLIF-Net

Pytorch code for "Sharing Leaky-Integrate-and-Fire Neurons for Memory-Efficient Spiking Neural Networks"

## Dependencies
* Python 3.9    
* PyTorch 1.10.0   
* Spikingjelly


## Training Details

For ResNet19/VGG16 architecture, 

(1) Standard SNN

```
python train_lifshare.py --dataset 'cifar10' --arch 'resnet19' --lifshare 'noshare' --batch_size 128 --learning_rate 1e-1
python train_lifshare.py --dataset 'cifar10' --arch 'vgg16' --lifshare 'noshare' --batch_size 128 --learning_rate 1e-1
```
(2) Cross-Layer sharing SNN
```
python train_lifshare.py --dataset 'cifar10' --arch 'resnet19' --lifshare 'layer' --batch_size 128 --learning_rate 1e-1
python train_lifshare.py --dataset 'cifar10' --arch 'vgg16' --lifshare 'layer' --batch_size 128 --learning_rate 1e-1
```
(3) Cross-Channel sharing SNN
```
python train_lifshare.py --dataset 'cifar10' --arch 'resnet19' --lifshare 'channel' --ch_group_num 2 --batch_size 128 --learning_rate 1e-1
python train_lifshare.py --dataset 'cifar10' --arch 'vgg16' --lifshare 'channel' --ch_group_num 2 --batch_size 128 --learning_rate 1e-1
```
(3) Cross-Layer+Channel sharing SNN
```
python train_lifshare.py --dataset 'cifar10' --arch 'resnet19' --lifshare 'layerchannel' --batch_size 128 --learning_rate 1e-1
python train_lifshare.py --dataset 'cifar10' --arch 'vgg16' --lifshare 'layerchannel' --batch_size 128 --learning_rate 1e-1
```
