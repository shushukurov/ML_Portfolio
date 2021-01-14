## Image Captioning (RNN, LSTM, Attention)

Generally, a captioning model is a combination of two separate architecture that is CNN (Convolutional Neural Networks)& RNN (Recurrent Neural Networks) and in this case LSTM (Long Short Term Memory), which is a special kind of RNN that includes a memory cell, in order to maintain the information for a longer period. 

## Recurrent Neural Network
Recurrent Neural Network (RNN) language models for image captioning.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/RNN.png" width=404 height=404>

## Long-Short-Term-Memory
Many people use a variant on the vanilla RNN called Long-Short Term Memory (LSTM) RNNs because Vanilla RNNs can be tough to train on long sequences due to vanishing and exploding gradients caused by repeated matrix multiplication. LSTMs solve this problem by replacing the simple update rule of the vanilla RNN with a gating mechanism as follows.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/LSTM.png" width=404 height=404>

## Attention
Attention is the idea of freeing the encoder-decoder architecture from the fixed-length internal representation.
This is achieved by keeping the intermediate outputs from the encoder LSTM from each step of the input sequence and training the model to learn to pay selective attention to these inputs and relate them to items in the output sequence.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/Attention.png" width=691 height=404>
