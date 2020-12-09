# DVC Phoneme Classifier
[Return to Home](index.md)
## Contents:
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction
The introduction goes here

blah

blah

blah

## Architecture
The model architecure is based on the neural network used by DeepSpeech 2, consisting of two one-dimensional convolutional layers, each followed by batch normalization and ReLU activation. After those layers, the model has three bidirectional GRU layers, then a fully-connected layer. All layers use 512 channels. The output of the network is a 71-element vector where each element corresponds to one of 71 possible phonemes, which includes silence and noise. We use a cross-entropy loss function between the predicted phoneme and ground truth phonemes to train the network.

## Data
Description of the data goes here

blah

blah

blah

## Training
Description of training process goes here

blah

blah

blah

## Evaluation
Description of evaluation process goes here

blah

blah

blah

## Results
Analysis of results goes here

#### Confusion Matrices
|  &nbsp;  |  Mixed testing data  |  Target testing data  |
|:--------:|:------------:|:-------------:|

|Model trained with mixed data|[![alt text](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)|![alt text](test_results/mixedmodel-targetdata/percent_confusion_matrix.png)|
|Model trained with target data|![alt text](test_results/targetmodel-mixeddata/percent_confusion_matrix.png)|![alt text](test_results/targetmodel-targetdata/percent_confusion_matrix.png)|

Here, we show the confusion matrices. Along the horizontal axis, we have the actual phoneme class. Along the vertical axis, we have the predicted phoneme class. The confusion matrix was computed by counting each predicted-actual phoneme combination, then dividing by the total number of times the actual phoneme appears. Thus, each cell's color represents a percentage of predicted phoneme for each actual phoneme. The color scale goes from purple to green too yellow, where a yellow cell means that when the actual phoneme was the phoneme associated with that column, the predicted phoneme was very often the phoneme associated with that row.

## Conclusion
Conclusion goes here
