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
The model architecure is based on the neural network used by DeepSpeech 2, consisting of two one-dimensional convolutional layers, each followed by batch normalization and ReLU activation. After those layers, the model has three bidirectional GRU layers, then a fully-connected layer. All layers use 512 channels. The output of the network is a 71-element vector where each element corresponds to one of 71 possible phonemes, which includes silence and noise.

We use a cross-entropy loss function between the predicted phoneme and ground truth phonemes to train the network. The cross-entropy loss is weighted to decrease the value of predicted silences, because silences are far more common than any other phoneme. The weights discourage the network from learning to always predict silence.

## Data
We use the Mozilla Common Voice Dataset for all experiments. This dataset consists of over 1,400 hours of English speech and corresponding transcripts from over 60,000 speakers, with accent labels for at least 700 hours of data. Details about the Mozilla Common Voice Dataset can be found [here](https://commonvoice.mozilla.org/en/datasets). We filter out all data that does not contain an accent label. Instead of using all accent categories, we use a binary accent labeling of whether the accent is or is not United States English. This is more broad than our intended Midwestern American English target accent, but will suffice for now given the lack of fine-grained accent annotations.

We extract ground truth phonemes using the Montreal Forced Aligner (MFA). MFA produces an alignment between the phonemes derived from the text and the speech audio, such that each phoneme is labeled with a start and end time at which it occurs in the audio. We one-hot encode the phoneme labels and upsample each one-hot vector to the number of 10 millisecond frames occupied by the phoneme. This means that each frame of the input spectrogram is aligned with the phoneme occurring during that frame. There are 71 phoneme categories, including silence and noise.

The data is partitioned into six groups:
1. 50 hours of only the target accent (American English) for training
2. 10 hours of only the target accent for validation
3. 10 hours of only the target accent for testing
4. 50 hours of mixed accents for training
5. 10 hours of mixed accents for validation
6. 10 hours of mixed accents for testing

We generate melspectrograms for all of the audio data, choosing a random five second chunk if the audio file is longer than five seconds. These melspectrograms serve as the input for our neural network.

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


|  &nbsp;  |  Mixed testing data  |  Target testing data  |
|:--------:|:------------:|:-------------:|
|Model trained with mixed data|[![alt text](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)|[![alt text](test_results/mixedmodel-targetdata/percent_confusion_matrix.png)](test_results/mixedmodel-targetdata/percent_confusion_matrix.png)|
|Model trained with target data|[![alt text](test_results/targetmodel-mixeddata/percent_confusion_matrix.png)](test_results/targetmodel-mixeddata/percent_confusion_matrix.png)|[![alt text](test_results/targetmodel-targetdata/percent_confusion_matrix.png)](test_results/targetmodel-targetdata/percent_confusion_matrix.png)|

Here, we show the confusion matrices. Along the horizontal axis, we have the actual phoneme class. Along the vertical axis, we have the predicted phoneme class. The confusion matrix was computed by counting each predicted-actual phoneme combination, then dividing by the total number of times the actual phoneme appears. Thus, each cell's color represents a percentage of predicted phoneme for each actual phoneme. The color scale goes from purple to green too yellow, where a yellow cell means that when the actual phoneme was the phoneme associated with that column, the predicted phoneme was very often the phoneme associated with that row.

## Conclusion
Conclusion goes here
