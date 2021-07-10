# ML8-ImageCaptioning
## Overview
Image captioning using attention based encoder-decoder model.The idea is discussed in [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Recurrent Neural Networks (RNN) are used for varied number of applications including machine translation. The Encoder-Decoder architecture is utilized for such settings where a varied-length input sequence is mapped to the varied-length output sequence. The same network can also be used for image captioning. We used a ResNet with pretrained weights as encoder to make feature vectors from the input images and GRU an variant of RNN as decoder.

## Implementation
In the image_captioning_.ipynb all of the preprocessing training and evaluation takes place.
- **Dataset Used:** MS-COCO(subset containing 6000 randomly shuffled images)
- **Vocabulary:** The vocabulary consists of mapping between words and indices(we limited the size of vocabulary to 5000 instead of 10000 as discussed in paper to save memory)
- **Encoder:** ResNet without the final classification layer with pretrained weights. we could also try trainig the encoder instead of loading pretrained weights.
- **Decoder:** GRU(Gated recurrent unit) is used as decoder with [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf). Using attention based architechture we can observe which parts of images were identified for generating words(or captions).
- **Caption Generation:** Based on highest probability instead of beam search.
- **Training:** Teacher forcing is used to reduce training time for the RNN.
- **Score:** Mean cosine similarity between the 5 true captions and the predicted caption.  


The prediciton.ipynb and prediction_final_for_external_images.ipynb will be used for loading the weights and models for direct prediciton of new images.
