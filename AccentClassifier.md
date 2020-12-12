# DVC Accent Classifier
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
This part of the project is the accent classifier which  is a convolutional neural network that takes in input audio and classifies the speech as either an American English or non-American English accent. The goal of this is to see
if it is possible to locate the certain sounds to find what the network distignuishes as non-American English and to provide that as input to the user. We trained our model initially on the Speech Accent Archive and later the Mozilla Common Voice dataset.

## Architecture
The accent classifier is a feed-forward convolutional neural network that contains an attention module in between an encoder and decoder. The encoder contains 4 convolutional blocks where each block has a 1D convolutional layer followed by batch normalization, ReLU, and dropout. The decoder is another 4 convolutional blocks with the same structure mentioned before and followed by two linear layers. These final linear layers will output the probability that the audio is American English accent using the melspectogram. We also tested an architecture for training where the attention module was placed after 8 convolutional blocks when training initially with the Speech Accent Archive.

## Data
For our initial training, we used the Speech Accent Archive provided here: https://www.kaggle.com/rtatman/speech-accent-archive. The dataset has 1GB MP3 files with 214 accents from 177 countries. All speakers speak the same sentences which are approximately 20-40 seconds long. These MP3 files are labeled through their filenames which contains the accent name and the corresponding accent number. The data has an associated CSV that gives information such as speaker id, origin, and location.
The target accent (American English) consists of 80% of the Speech Accent Acrhive dataset.

After our inital inconclusive results for the training on the Speech Accent Archive, we transitioned into using the Mozilla Common Voice Dataset found here: https://commonvoice.mozilla.org/en/datasets. 
This dataset includes 1,400 hours of validated English speech and transcripts by 60,000+ speakers, with accent labels for 700+ hours. We will only use labeled data, and we simplify the classification to binary labels of American English and not American English. The data is structured in a CSV file that includes the relative path to the audio file, transcript, age, gender, and accent. The audio files are provided in the MP3 format. We will preprocess this data using the Montreal Forced Aligner (MFA) to extract ground truth phonemes with start and end time.
The target accent (American English) consists of 55% of the Mozilla Common Voice dataset.
## Training
The preprocessing for accent classification will turn each audio file into log-melspectograms with a sample rate of 16000. The model will train using randomly selected second chunks of these melspectograms that are collated to create the batch. The training was conducted on 1, 2, and 4 second chunks with their results seen below (Figure 1). The target accent was American English and given a value of 1. During training, validation will occur once per training epoch and accuracy, precision, loss, and recall will all be tracked through both training and validation epochs. Additionally, the model will use binary cross entropy for it's loss function and will use an Adam optimization during training.


Our initial experiments with the Speech Accent Archive were based around finding the best chunk sized for training and whether attention helped improve the performance of the model. 


We tested if using attention or no attention would provide better performance for the model. 
We concluded that the model with the Attention module had better overrall performance using the lowest validation loss. The graphs for these training runs can be seen below


| Model | Accuracy| Loss | 
|:-----:|:------:|:--------:|
|Model With Attention|.903| .259|
|Model Without Attention|.864|.283|

*This table shows the best loss during training for models  with  and  without  Attention.   It  also  shows  the  corresponding accuracy at that loss*

The results for the training runs for using chunk sizes of 1, 2, and 4 can be seen below using the Speech Accent Archive. 

| Chunk Size that Model is Trained on| Accuracy| Loss | 
|:-----:|:------:|:--------:|
|1 Second|.867| .363|
|2 Second|.906|.283|
|4 Second|.917|.236|

*This table shows the best loss during training for models that were trained on 1, 2, and 4 second chunks of audio.It also shows the corresponding accuracy at that loss*

We concluded that the two second chunk size had the best performance in terms of trade-off as the 4 second chunk size took much longer for training. Additionally, the results of using the attention module were slightly better than without. The precision, accuracy, recall, and loss for our final training with a chunk size of 2 seconds and the attention module trained on the Speech Accent Archive can be seen below. 
The target accent (American English) was 80% of the dataset.
[![alt text](images/accent-c/recall.png)](iimages/accent-c/recall.png)

*Loss (top left), Precision (top right), Accuracy (bottom left), and Recall (bottom right) for the final training run using the Speech Accent Archive with a chunk size of 2. This training run had a max epoch of 200. The target accent (American English) was 80 percent of the total dataset.*

We initially had a training run for the Mozilla Common Voice dataset with max epoch of 50. This run used both attention and 2-second chunk size and validation occurred at the end of every training epoch. The target accent (American English) was 55% of this dataset.
The associated tensorboard can be found here: [https://tensorboard.dev/experiment/ZvyEUeHaRMGXJQS86l1H1w/#scalars](https://tensorboard.dev/experiment/ZvyEUeHaRMGXJQS86l1H1w/#scalars)

[![alt text](images/accent-c/50epoch.png)](images/accent-c/50epoch.png)
*These four graphs show the accuracy and loss for the max epoch of 50 run using the Mozilla Common Voice Dataset. The top left is the accuracy over epochs for training, top right is accuracy over epochs for validation, bottom left is loss over epochs for training, and bottom right is loss over epochs for validation. 55 percent of the dataset was the target accent.*

[![alt text](https://i.imgur.com/AEfY3C1.png)](https://i.imgur.com/AEfY3C1.png)
*These two graphs show the precision and recall or the max epoch of 50 run using the Mozilla Common Voice Dataset. The top graph is for precision for both training and validation, and the bottom graph is for recall for both training and validation. The minimum loss occurred at 15. This checkpoint was used for later evaluation. 55 percent of the dataset was the target accent.*

After noticing that the model converged at around 15 epochs, we ran another run with max epoch at 20 to confirm this.
The target accent (American English) was 55% of this dataset.
The tensorboard for this can be found at: [https://tensorboard.dev/experiment/KbcwNHEDRDOHBYjLgNEOWg/#scalars](https://tensorboard.dev/experiment/KbcwNHEDRDOHBYjLgNEOWg/#scalars)

[![alt text](https://i.imgur.com/Lo7v36S.png)](https://i.imgur.com/Lo7v36S.png)
*These two graphs show the accuracy for the training run on the Mozilla Common Voice Dataset where the max epoch was 20*
[![alt text](https://i.imgur.com/Bq0pqyS.png)](https://i.imgur.com/Bq0pqyS.png)
*These two graphs show the loss for the training run on the Mozilla Common Voice Dataset where the max epoch was 20*
[![alt text](https://i.imgur.com/QUCFSW1.png)](https://i.imgur.com/QUCFSW1.png)
*These two graphs show the precision for the training run on the Mozilla Common Voice Dataset where the max epoch was 20*
[![alt text](https://i.imgur.com/BQ5P3gP.png)](https://i.imgur.com/BQ5P3gP.png)
*These two graphs show the recall for the training run on the Mozilla Common Voice Dataset where the max epoch was 20*




 As part of our experiment, we also conducted additional to test our hypothesis if the model would work better with choosing a non-American accent as our target accent due to the greater diversity within American English accent.
The target accent, Indian, consisted of only 11% of the dataset and we modified the model by giving weight to the target value. 
The tensorboard for this run can be found here: [https://tensorboard.dev/experiment/2Apz8Eq5T1Ky7qvnA8VLBw/#scalars](https://tensorboard.dev/experiment/2Apz8Eq5T1Ky7qvnA8VLBw/#scalars)
[![alt text](https://i.imgur.com/WUxEINh.png)](https://i.imgur.com/WUxEINh.png)
*These two graphs show the accuracy for the training run on the Mozilla Common Voice Dataset where the target accent was 'Indian'*

[![alt text](https://i.imgur.com/E0JXlVh.png)](https://i.imgur.com/E0JXlVh.png)
*These two graphs show the loss for the training run on the Mozilla Common Voice Dataset where the target accent was 'Indian'*

[![alt text](https://i.imgur.com/BbIoGad.png)](https://i.imgur.com/BbIoGad.png)
*These two graphs show the precision for the training run on the Mozilla Common Voice Dataset where the target accent was 'Indian'*

[![alt text](https://i.imgur.com/0uk94Ay.png)](https://i.imgur.com/0uk94Ay.png)
*These two graphs show the recall for the training run on the Mozilla Common Voice Dataset where the target accent was 'Indian'*


## Evaluation

For all the following evaluations, we used a checkpoint from the training run using the Mozilla Common Voice Dataset where the the max epochs was 50.

We used the attention weights to create visualizations of what the model was attending to. An example of this attention visualization can be seen below the axises of the module are frames by frames for a 2 second chunk of audio (approximately 201 frames). We concluded that where there are horizontal 'lines' on the visualization all the frames on the horizontal axis were attending to a single frame from the vertical axis. 
However, we could not retreive any conclusive results from these visualizations but did find that for our model that trained on the Mozilla Common Voice Dataset that the 'vertical' frames that was being attended were moving approxmiately by 50 frames for each visualization generated in sequential order signifying there was a pattern for the module. For instance, in the example below which are taken .5 seconds apart (hop size)) for the given audio , we can see that the lines move approximately 50 frames upward in the vertical direction across the 3 with the left occurring first in sequential order.

| Attention for 2 second chunk starting at 0 seconds| Attention for 2 second chunk starting at .5 seconds| Attention for 2 second chunk starting at 1 seconds | 
|:-----:|:------:|:--------:|
|[![alt text](https://i.imgur.com/g4t3WVw.png)](https://i.imgur.com/g4t3WVw.png)|[![alt text](https://i.imgur.com/MZEVm9Q.png)](https://i.imgur.com/MZEVm9Q.png)| [![alt text](https://i.imgur.com/J29zeeB.png)](https://i.imgur.com/J29zeeB.png)|

*These images are the visualizations for the attention a 1.5 audio clip that has an Australian accent. From left to right these visualizations go in sequential order for increments of .5 seconds and each is 2 seconds (201 frames).*


Due to the inconclusive evidence from the attention visualizations, we 
conducted some initial analysis by passing audio files into the model and retreiving the probablities 
that given two second chunks were American English. This was done with the max epoch of 50 training run doneThe audio file was converted into a mel-spectogram and two second chunks who's start position
with the Mozilla Common Voice Dataset
incremented by .5 seconds (converted to frames) were passed sequentially into the model. The graph generated was the probablities for the chunk plotted against where that 
chunk started. 

The first audio file that was  tested on was an American English Accent seen [here](audio/ShortEnglish.mp3).
The probablity over time graph for this given audio file is below (this can be expanded by clicking on the image):

[![alt text](https://i.imgur.com/ypIlWHF.png)](https://i.imgur.com/ypIlWHF.png)
*An annotated graph of the probablities against where the chunk starts for an audio file where the speaker speaks only English in an American accent. The annotations at the bottom are the speech in the audio file and are approxoimated for where each words begins and ends.*

When looking at this graph, we noticed that often hard consonants and long vowels were more associated with American English. 
Additionally, we found that for the majority of the audio the model associated a probability of greater than .5 and the average of the probability for the chunks was .63. Although the average was not as high as it should be considering that it was spoken by a native english speaker, it did show the model had some success in correctly classifying.

[Here](audio/TeluguTrue.mp3) is an audio file of a speech spoken in a non-English Language (Telugu). The probablity over time graph for this given audio file is below:

[![alt text](https://i.imgur.com/jgyXSCo.png)](https://i.imgur.com/jgyXSCo.png)
*This is a graph of the proabablities for each chunk against where the chunk starts for an audio file where the speaker speaks Telugu (non-English language).*

For this graph the average probabliity was .47 which again does the ideal results we want for the probablity, but the sharp drops in probability within the graphd eos establish that it is able to distinguish to an extent between American English and not English.

W also tested transitioning from an American English Accent to French in the same audio clip. The audio can be found [here](audio/EngFrench.mp3).
The probablity over time graph for this given audio file is below:

[![alt text](https://i.imgur.com/qBsYcS2.png)](https://i.imgur.com/qBsYcS2.png)
*This is a graph of the proabablities for each chunk against where the chunk starts for an audio file where the speaker starts by speaking in an American English accent and transitions to speaking French at the 21 second mark*

This examples switches languages at the 21 second mark from American English to French. Here, we can see a drop off in probability that indicates that it was able to tell the change of language. The average probability for the first 20 seconds (American English portion) of the audio is .62 while the last 30 seconds (or the French portion) had an average probability of .45. Once again this large change in probability indciates that the language
did to an extent learn the differences between the accent. However, due to the number of similiar phonemes in the languages could be the reason that there is not a larger
difference between the two probabilties. 




## Conclusion
After analyzing these graphs, we concluded the probabilities generally lowered when speaking non-American English. We found that due to the number of examples we analyzed that the model was able to distinguished between the accents to an extent from the average probability change. 
Additionally, by looking at the validation graphs seen in the training runs on the Mozilla Common Voice Dataset,
 the precision and recall highlighted that it was able to classify many samples accurately to a statistically significant margin.
We belive that the manual evaluation and validation grpahs showcase that this method and prodcedure has potential as a way to idnetify misprounciation due 
the models ability to distinguish accents. However, due to number of similiar sounds in many languages, the process for further
expanding the model's ability to distinguish might be difficult or need an alternate approach.