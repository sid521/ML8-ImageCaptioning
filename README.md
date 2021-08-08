# ML8-ImageCaptioning/Video Summarizer
This project is created as part of IITISoC'21 by Team ML8: [Aditya Gouroju](https://github.com/StrawHat369), [Potu Sidhartha Reddy](https://github.com/sid521), [Pepeti Venkata Sai Kesava Siddhardha](https://github.com/pepetikesavasiddhardha)

Mentors - Aryan Rastogi, Bharat Gupta, Sakshee Patil, Kashish Bansal
## Overview
Image captioning using attention based encoder-decoder model.The idea is discussed in [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Recurrent Neural Networks (RNN) are used for varied number of applications including machine translation. The Encoder-Decoder architecture is utilized for such settings where a varied-length input sequence is mapped to the varied-length output sequence. The same network can also be used for image captioning. We used a ResNet with pretrained weights as encoder to make feature vectors from the input images and GRU an variant of RNN as decoder.

Now for Video Summarization using OpenCV library we will capture frames in video at specific time interval(1 frame per 2 seconds) and we will generate captions to all these frames using above said Image captioning model and retain only those captions which have a low similarity score with the immediate previous caption and that Threshold similarity score is 0.5.Then we perform Abstractive Summarization using T5 base Transformer model
## Dependencies
```
python 3.7.11
pandas==1.1.5
numpy==1.19.5
scikit-learn==0.22.2.post1
opencv-python=4.1.2.30
matplotlib==3.2.2
tensorflow==2.5.0
keras==2.4.3
keras-Preprocessing==1.1.2
pip==21.1.3
scipy ==1.4.1
tqdm==4.41.1
sentence-transformers==2.0.0
transformers==4.9.1
```
## Implementation
In the image_captioning_.ipynb we download the datasets and all of the preprocessing training and evaluation takes place.
- **Dataset Used:** MS-COCO(subset containing 15000 randomly shuffled images)
- **Vocabulary:** The vocabulary consists of mapping between words and indices(we limited the size of vocabulary to 5000 instead of 10000 as discussed in paper to save memory)
- **Encoder:** ResNet without the final classification layer with pretrained weights. we could also try trainig the encoder instead of loading pretrained weights.
- **Decoder:** GRU(Gated recurrent unit) is used as decoder with [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf). Using attention based architechture we can observe which parts of images were identified for generating words(or captions). 2 GRUs are stacked on top of each other and 3 fully connected layers for predictions with 0.25 droupout at every stage in decoder.
- **Caption Generation:** Based on highest probability/greedy search.
- **Training:** Teacher forcing is used to reduce training time for the RNN.
- **Score:** Maximum cosine similarity between the 5 true captions and the predicted caption. Mean cosine similarity of 50 random images :  0.82622829
- **Video to frames:** Using OpenCV
- **Transformer used(for Summarization):** T5 base

**Hyperparameters involved in Image captioning:**
Hyper parameter| Value
-------------  | -------------
Embedding size |   256
Vocabulary size|   5001
Batch Size     |   64
GRU 1 Output   |   512
GRU 2 Output   |   512
FC1 units      |   512
FC2 units      |   512
FC3 units      |   5001
Dropout        |   0.25

**Hyperparameters involved in Video Summarization:**
Hyper parameter| Value
-------------  | -------------
max_length     |   512
min_length     |   50
length_penalty |   2.0
num_beams      |   4
early_stopping |   True

The prediction_beam.ipynb will be used for loading the weights and models for direct prediciton of new images and videos. You can find the weights,resnet model and tokenizer trained on taking random 50k subset images [here](https://drive.google.com/drive/folders/1f_G7w1mrYfBLKkxIFuy9xfWzn48Y32nx?usp=sharing)

## Results
   ![ss2](https://user-images.githubusercontent.com/70747076/128628236-10af847c-fef1-4de2-bcc6-4f1059dacc89.jpeg)
   ![final_prediction](https://user-images.githubusercontent.com/70747076/128628189-21c8627b-0380-4ad7-a2aa-1cdcff146212.jpeg)

## Future Enhancements
- [Audio Visual Scene-Aware Dialog](https://openaccess.thecvf.com/content_CVPR_2019/papers/Alamri_Audio_Visual_Scene-Aware_Dialog_CVPR_2019_paper.pdf)

## References
- [ Image captioning with visual attention ](https://www.tensorflow.org/tutorials/text/image_captioning)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Lecture 8:Machine Translation,Sequence-to-sequence and Attention CS224N](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)
- [LSTM is dead. Long Live Transformers!](https://youtu.be/S27pHKBEp30)
- [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://youtu.be/4Bdc55j80l8)
- [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
