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
The experiments use the Mozilla Common Voice Dataset \cite{ardila2019common}, found [here](https://commonvoice.mozilla.org/en/datasets). This dataset includes 1,400 hours of validated English speech and transcripts by 60,000+ speakers, with accent labels for 700+ hours. We will only use labeled data, and we simplify the classification to binary labels of American English and not American English. The data is structured in a CSV file that includes the relative path to the audio file, transcript, age, gender, and accent. The audio files are provided in the MP3 format.

The data we used was selected randomoly from the Mozilla Common Voice Dataset and split into six disjoint partitions:
1. 50 hours of only the target accent (American English) for training
2. 10 hours of only the target accent for validation
3. 10 hours of only the target accent for testing
4. 50 hours of mixed accents for training
5. 10 hours of mixed accents for validation
6. 10 hours of mixed accents for testing

Audio clips are downsampled to 16 kHz and preprocessed to melspectrograms with 80 Mel-frequency bins with 1024 samples per window and 160 samples per hop. We generate melspectrograms for all of the audio data, choosing a random five second chunk if the audio file is longer than five seconds. These melspectrograms serve as the input for our neural network. The melspectrogram shows the frequencies for a particular 10 millisecond frame, and the model predicts a phoneme for each frame of the melspectrogram.

We extract ground truth phonemes using the Montreal Forced Aligner (MFA). MFA produces an alignment between the phonemes derived from the text and the speech audio, such that each phoneme is labeled with a start and end time at which it occurs in the audio. We one-hot encode the phoneme labels and upsample each one-hot vector to the number of 10 millisecond frames occupied by the phoneme. This means that each frame of the input spectrogram is aligned with the phoneme occurring during that frame. There are 71 phoneme categories, including silence and noise.

## Training
We trained two models, one using the target accent training and validation partitions, and another using the mixed accent training and validation partitions. The training is completed with a gradient clipping value of 0.5 and an Adam optimizer with a learning rate of 1e-3. We trained the models for 100 epochs.

We used gradient clipping due to a significant stabilization in the training loss and accuracy. Below, we show the graphs for training with and without gradient clipping using the target accent data.

(insert graphs here)

When we used gradient clipping, the learning appears to always start very slowly. During this time, the network almost always predicts a small handful of phonemes, especially silence. We hypothesize that the weights are slowly growing during this time until the much faster learning begins. For both models, the validation accuracy and loss becomes very unstable at this point. This may be due to having a validation dataset that is too small, and future training with a larger validation set is planned. In general, the model trained on only the target accent is able to achieve a higher best accuracy.

(insert more graphs here)

## Evaluation
To evaluate the model, we tested both the target-trained and mixed-trained models against both the target accent and mixed accent testing data. We trained using the saved model weights at the point with the lowest validation loss from training. Quantitatively, we look at the performance of each model on the target and mixed testing data. We investigate whether the target-trained model performs better on the target accent testing data compared to the mixed accent testing data. We also investigate how target-trained model's performance gap on the target and mixed accent testing data compares to the mixed-model's performance gap. A larger performance gap might suggest that the target-trained model has learned phoneme nuances that are specific to the target accent by training exclusively with the target accent.

Qualitatively, we manually inspect a random few audio files from the test set. For these, we plot the model's confidence for each frame in the audio file. For each frame of the melspectrogram input, the network outputs a 71-element vector where each value corresponds to how much the model thinks the frame is that particular phoneme. We take a softmax of this vector, then choose the maximum softmaxed value to represent the confidence on that particular frame. We hope to see lower confidence in frames where accent errors occur for the target-trained model.

## Results

|  &nbsp;  |  Mixed testing data  |  Target testing data  |
|:--------:|:------------:|:-------------:|
|Mixed-trained model|[![alt text](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)](test_results/mixedmodel-mixeddata/percent_confusion_matrix.png)|[![alt text](test_results/mixedmodel-targetdata/percent_confusion_matrix.png)](test_results/mixedmodel-targetdata/percent_confusion_matrix.png)|
|Target-trained model|[![alt text](test_results/targetmodel-mixeddata/percent_confusion_matrix.png)](test_results/targetmodel-mixeddata/percent_confusion_matrix.png)|[![alt text](test_results/targetmodel-targetdata/percent_confusion_matrix.png)](test_results/targetmodel-targetdata/percent_confusion_matrix.png)|

Here, we show the confusion matrices. Along the horizontal axis, we have the actual phoneme class. Along the vertical axis, we have the predicted phoneme class. The confusion matrix was computed by counting each predicted-actual phoneme combination, then dividing by the total number of times the actual phoneme appears. Thus, each cell's color represents a percentage of predicted phoneme for each actual phoneme. The color scale goes from purple to green too yellow, where a yellow cell means that when the actual phoneme was the phoneme associated with that column, the predicted phoneme was very often the phoneme associated with that row.

| Model | Accent | Accuracy | Loss |
|:-----:|:------:|:--------:|:----:|
|Mixed-trained|Mixed|0.442|2.471|
|Mixed-trained|Target|0.455|2.399|
|Target-trained|Mixed|0.504|2.230|
|Target-trained|Target|0.534|2.033|

Here, we show the average testing accuracy and loss for both models with both testing data partitions. While both models improved slightly with the target accent testing data, the target-trained model had a slightly larger accuracy performance gap of 3%, compared to the mixed-trained model's 1.3% performance gap. In general, the target-trained model performed better than the mixed-trained model, even on the mixed accent testing data. This may be due to a larger proportion of American English accents relative to other accents in the mixed accent testing data.

## Conclusion
Conclusion goes here
