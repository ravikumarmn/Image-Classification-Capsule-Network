**What is a Capsule**

> A Capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part.

## Requirements
- Python 3
  - Tested with version 3.11.7


## Usage

### Training and Evaluation
**Step 1.**
Clone this repository with ``git`` and install project dependencies.

```bash
$ pip install -r requirements.txt
```

**Step 2.** 
Start the CapsNet on MNIST training and evaluation:

- Training with default settings:
```bash
$ python main.py
```

- Training on 8 GPUs with 30 epochs and 1 routing iteration:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --epochs 30 --num-routing 1 --threads 16 --batch-size 128 --test-batch-size 128
```


### The Default Hyper Parameters

| Parameter | Value | CLI arguments |
| --- | --- | --- |
| Training epochs | 10 | --epochs 10 |
| Learning rate | 0.01 | --lr 0.01 |
| Training batch size | 128 | --batch-size 128 |
| Testing batch size | 128 | --test-batch-size 128 |
| Log interval | 10 | --log-interval 10 |
| Disables CUDA training | false | --no-cuda |
| Num. of channels produced by the convolution | 256 | --num-conv-out-channel 256 |
| Num. of input channels to the convolution | 1 | --num-conv-in-channel 1 |
| Num. of primary unit | 8 | --num-primary-unit 8 |
| Primary unit size | 1152 | --primary-unit-size 1152 |
| Num. of digit classes | 10 | --num-classes 10 |
| Output unit size | 16 | --output-unit-size 16 |
| Num. routing iteration | 3 | --num-routing 3 |
| Use reconstruction loss | true | --use-reconstruction-loss |
| Regularization coefficient for reconstruction loss | 0.0005 | --regularization-scale 0.0005 |
| Dataset name (mnist, cifar10) | mnist | --dataset mnist |
| Input image width to the convolution | 28 | --input-width 28 |
| Input image height to the convolution | 28 | --input-height 28 |


### Training Loss and Accuracy

The training losses and accuracies for CapsNet-v4 (50 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/train_loss_accuracy.png)

Training accuracy. Highest training accuracy: 100%

![](results/train_accuracy.png)

Training loss. Lowest training error: 0.1938%

![](results/train_loss.png)

### Test Loss and Accuracy

The test losses and accuracies for CapsNet-v4 (50 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/test_loss_accuracy.png)

Test accuracy. Highest test accuracy: 99.32%

![](results/test_accuracy.png)

Test loss. Lowest test error: 0.2002%

![](results/test_loss.png)


![](results/training_speed.png)

In my case, these are the hyperparameters I used for the training setup:

- batch size: 128
- Epochs: 50
- Num. of routing: 3
- Use reconstruction loss: yes
- Regularization scale for reconstruction loss: 0.0005


Total number of parameters on (with reconstruction network): 7302160 (7 million)
```

