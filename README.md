## Table of contents
* [Introduction](#introduction)
* [Methods](#methods)
  * [Image Classification](#image-classification)
  * [VGG16](#vgg16)
  * [Datasets](#datasets)
  * [Data Augmentation](#data-augmentation)
  * [Training](#training)
* [Setup](#setup)
* [Technologies](#technologies)

## Introduction
What is transfer learning anyways? Transfer learning is a machine learning technique used to transfer knowledge from one model to another. We can make a relatively smart model with little training data.
This project implements transfer learning in the task of classifying American Sign Language (ASL) 
hand gestures. Using VGG16 as the pre-trained model and comparing it to a custom CNN model, the goal is to evaluate how transfer learning improves model performance, 
especially when data and computational resources are limited.

## Methods

### Image Classification
Image classification is a method where (in this case) we use convolutional neural networks (CNNs) to group images. We use the neural network to figure out which letter is being held up in sign language. 
We focus on leveraging the VGG16 model architecture for transfer learning and compare its performance with a custom CNN.

### VGG16
VGG16 is a pre-trained convolutional neural network designed for the ImageNet dataset. The architecture involves a series of convolutional and pooling layers, ultimately creating a feature map that is later fed into fully connected layers for classification.

### Datasets
- **ImageNet**: Has 14 million images for 1000 classes. Its pre-trained model is used for transfer learning.
- **ASL Alphabet Test**: Has 870 images of ASL hand gestures across 29 classes. I've split 70:20:10 into training, validation, and test sets. This is actually a pretty small amount of data, this will put pressure on transfer learning to provide usable results.

### Data Augmentation
To tackle the limited size of the ASL dataset, data augmentation techniques like flipping, rotating, zooming, adjusting contrast, and translating are applied to create a model capable of generalizing to unseen data.

### Training
The VGG16 model is fine-tuned with our dataset by freezing the convolutional layers and only training the newly added dense layers. The custom CNN is trained from scratch without transfer learning for comparison (it's bad).

## Results

### Fine-tuned VGG16 Model
- **Accuracy**: ~69% on the test set.
- **Training Time**: 28 minutes.
- **Performance**: Demonstrated strong generalization to unseen data, significantly outperforming the custom CNN model.

### CNN Without Transfer Learning
- **Accuracy**: ~6% on the test set.
- **Performance**: The custom CNN performed poorly, highlighting the benefits of transfer learning.

### Transfer Learning Model Without Data Augmentation
- **Accuracy**: ~49% on the test set.
- **Performance**: Data augmentation alone boosted accuracy by 21%, reducing overfitting compared to the model without augmentation.

## Setup
1. Clone the repository for the ImageNet model:
   ```bash
   git clone https://github.com/hukenovs/hagrid
   ```
2. Install dependencies:
   ```bash
   pip install -r hagrid/requirements.txt
   ```

## Technologies
- Python 3.9.9
- TensorFlow 2.x
- Keras 2.x
- scikit-learn 1.0.2
- pandas 1.3.5
- numpy 1.21.5
- matplotlib 3.4.3
- seaborn 0.11.2
