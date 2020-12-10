---
title: Deep Voice Coach
---
Authors:
- Abhinav Reddy (AbhinavReddy2022@u.northwestern.edu)
- Caedon Hsieh (CaedonHsieh2022@u.northwestern.edu)

University: Northwestern University

Course: COMP_SCI 396/496: Deep Learning

Professor: Bryan Pardo

Acknowledgements: We want to thank Max Morrison for his help, vision, guidance, and time, which made this project possible. We also want to thank Bryan Pardo and the Interactive Audio Lab for equipping us with the tools necessary for this project.

Deep Voice Coach (DVC) aims to provide an interactive voice coach that teaches users to speak with a target accent. In practice, DVC could assist in accent-learning for actors and actresses and help new or non-native language learners improve their American English accent and pronunciation. Existing accent learning systems are either impractical or insufficient. Many platforms only provide small snippets of different accents. Few tools provide recording and playback functionality, and none of them give real-time feedback on recordings. Personal voice and accent coaches are an alternative to online systems, but they are expensive and not easily accessible for most language learners. Ideally, DVC will be an accessible accent learning system that provides real-time feedback for users.

We built a two-part deep learning system that classifies accents in speech to provide feedback that helps users learn a particular accent. For this project, we focus on learning an American English accent. DVC explores the accent learning problem through two deep learning sub-problems: accent classification and phoneme classification. The accent classifier takes in a speech audio file and attempts to classify the speech as a boolean American English or not by taking in an audio file prepossessed into a melspectogram and giving confidence scores for sequential audio chunks. The phoneme classifier takes in a similarly preprocessed speech audio file and attempts to classify each phoneme, giving a confidence score where a lower confidence score indicates a less American English accent on that particular phoneme.

- **[Click here to learn more about the Accent Classifier](AccentClassifier.md)**, which uses an attention-based neural network to classify speech as either the target accent or not the target accent
- **[Click here to learn more about the Phoneme Classifier](PhonemeClassifier.md)**, which attempts to classify phonemes in given speech, extracting perceived accent errors from the classifier's confidence

We trained and tested both of these deep learning parts with audio speech data from the Speech Accent Archive and Mozilla Common Voice datasets, achieving mixed results. The accent classifier's overall confidence drops when switching from American English to not American English, and the phoneme classifier trained on only the target accent had a larger confidence gap with target and mixed accent test data compared to the model trained on mixed accents. However, the results are not conclusive enough for practical usability. Future work may include exploring different data or more data, attempting to achieve more stable results.
