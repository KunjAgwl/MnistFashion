**Training Resnet model to recognise Mnist Fashion dataset**

## Why use Resnet??

So i have been learning about different architecture of YOLO models. Some versions of Yolo use Resnet as part of its architecture so  I wanted to experiment with it . There are still few things i dont understand about Resnet architecture . I will still try my best to explain the architecture.

## Why Resnet was Invented

<img width="1043" height="367" alt="image" src="https://github.com/user-attachments/assets/809485d7-45eb-43d5-8183-5a9d25048e3e" />

Suppose you want to build a CNN network for a really complex recognition task. You try to increase the number of layer but if they are too many number of layers a new probem arises that is vanishing or exploding gradient problem . Resnet was invented to handle more number of layers bypassing this problem using skip connections method.

<img width="803" height="367" alt="image" src="https://github.com/user-attachments/assets/25e8acdc-7a3b-424e-8810-039589bc614a" />


# Residual Layers Structure

Each layer consists of multiple BasicBlocks stacked together using the `_make_layer` function:

## Layer 1

'''self.layer1 = self._make_layer(block, 64, num_blocks, stride=1)'''

- **Channels:** 64 → 64 (no channel change)
- **Spatial size:** 28×28 → 28×28 (stride=1, no downsampling)
- **Blocks:** num_blocks BasicBlocks
- **Purpose:** Feature refinement at full resolution

## Layer 2
self.layer2 = self._make_layer(block, 128, num_blocks, stride=2)

- **Channels:** 64 → 128 (doubles channels)
- **Spatial size:** 28×28 → 14×14 (stride=2, first downsampling)
- **Blocks:** num_blocks BasicBlocks
- **Purpose:** Extract higher-level features at reduced resolution

## Layer 3
self.layer3 = self._make_layer(block, 256, num_blocks, stride=2)

- **Channels:** 128 → 256 (doubles again)
- **Spatial size:** 14×14 → 7×7 (second downsampling)
- **Blocks:** num_blocks BasicBlocks
- **Purpose:** More abstract feature extraction

## Layer 4

self.layer4 = self._make_layer(block, 512, num_blocks, stride=2)

- **Channels:** 256 → 512 (final doubling)
- **Spatial size:** 7×7 → 4×4 (final downsampling)
- **Blocks:** num_blocks BasicBlocks
- **Purpose:** High-level semantic features

## Final Classification

self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(512 * block.expansion, num_classes)

- **Global Average Pooling:** 4×4×512 → 1×1×512
- **Flatten:** 1×1×512 → 512
- **Fully Connected:** 512 → 10 (for MNIST's 10 classes)



