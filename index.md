---
title: Deep Voice Coach
---
Authors:
- Abhinav Reddy (AbhinavReddy2022@u.northwestern.edu)
- Caedon Hsieh (CaedonHsieh2022@u.northwestern.edu)

University: Northwestern University

Course: COMP_SCI 396/496: Deep Learning

Professor: Bryan Pardo

Research for: Interactive Audio Lab

Deep Voice Coach (DVC) aims to provide an interactive voice coach that teaches users to speak with a target accent. In this report, we explore two ways to handle accent classification using deep learning, seeking to identify errors where the user speech does not match the target accent. These two methods include:
1. An [Accent Classifier](AccentClassifier.md) that uses an attention-based neural network to classify speech as either the target accent or not the target accent
2. A [Phoneme Classifier](PhonemeClassifier.md) that attempts to classify phonemes in given speech, extracting perceived accent errors from the classifier's confidence

(further elaborate on the high level motivations here)

(add acknowledgements somewhere - Max!)
