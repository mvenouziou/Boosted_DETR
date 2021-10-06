# DETR for Tensorflow

This is my implementation of the DETR object detector in Tensorflow. It has been coded from first principles as presented in the paper [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 

#### Notes:

- My model is written completely within the Tensorflow 2 / Keras subclass API and should be easy for anyone familiar with that API to train, modify and customize to their task. 

- It requires only standard dependencies used in most Tensorflow projects, plus a single Scipy function, *linear_sum_assignment*, for bipartite matching. 

- The loss function, metrics, training regime and text tokenization / de-tokenization are all built in. Train by passing an optimizer into model.compile() and then using model.fit() as usual.

- I modified the standard DETR architecture to accept both classes (exactly one per object) and subclasses (0 or more per object), allowing the option for fine-grained object descriptions. Panoptic segmentation has not yet been implemented. 


####  Training

- The official DETR model was intensively trained on 16 high-end GPU's for 3 full days, and neither my computer's integrated Intel graphics nor Google Colab's single GPU are exactly up the task. Unfortunately the model's reliance on reliance on Scipy's *linear_sum_assignment* for bipatite matching is not compatible with Colab TPU training due my . (The official DETR github also relies on this function.)

- Flush with $300 in free credits from Google Cloud, I fully prepared the model for training on their platform only to discover GPU's are specifically excluded from the offer. I am currently searching for an alternative cloud system to train the model at a reasonable price.

#### Comarisons:
- Although I did **not** make use of their repository, it is worth citing the [official Github Repo](https://github.com/facebookresearch/detr/tree/master) PyTorch implementation, publicly available under Apache License. I did look through their repo to see if they used an alternative to *scipy.optimize.linear_sum_assignment* for performing bipartite matching. (They did not.) 

- My model appears to match the official DETR PyTorch implementation's inference and training speeds.

- This implemenatation is in stark contrast to the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), which does not use the standard TF / Keras API for training and customization. Perhaps a deeper dive into their code base will bring clarity for this choice. It appears they are in the process of significantly simplifying their code, but training and model building is still conducted outside standard Tensorflow practices.)

----

## An Idea

The bipartite matching algorithm (*linear_sum_assignment*) speed seems to be the main limiting factor to naively applying DETR's training regime to traditional object detectors. Assignments involving tens of thousands of proposals may add several seconds / batch during training. 

It would be interesting to experiment with a modified CNN training routine where it is fully trained as normal, then training an additional "sifting" layer using bipartite matching to replace NMS in weeding out predictions.

----

##  Background

After Google and others showed great success in Natural Language Processing with Attention Transformers (developed in the paper [*Attention is All You Need*](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (2017)), researchers quickly began adapting Transformers to image processing tasks. [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) (2020) a.k.a. DETR is a novel extension of Attention Transformers to Object Detection.

#### Transformers in NLP

RNN's with Attention had become the defacto standard for NLP tasks. These, however, are hindered by lack of training parallelism and slowdowns on large dimensions and very long sequences. *Attention is All You Need* replaced RNN's entirely with a modified attention architecture ("Transformers") utilizing joint encoder-decoder attention and decoder self-attention. Their implementation has all the benefits of attention (allowing the decoder to selectively use the full set of encoder vectors) but can be trained in parallel and uses linear projections onto low-dimensional subspaces to provide more nuanced encoder-decoder interations at reduced computational costs.

#### Transformers in Image Captioning

State of the Art Image processing using Convolutional Neural Networks do not suffer from NLP's previous lack of parallelism, removing one of the main benefits of switching to transformers. In addition, transformers have an enormous memory footprint compared to CNNs, and they lack CNNs natural sense of spacial positioning. 

The most likely benefit of transformers would be for tasks involving both image processing and NLP, such as imaging captioning. [*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*](https://proceedings.mlr.press/v37/xuc15.pdf) (2015) had already used a hybrid approach connecting a CNN encoder with an RNN decoder. Replacing the RNN with Transformers as a natural extension.

*(See my project [Attention is What You Get](https://github.com/mvenouziou/Project-Attention-Is-What-You-Get) where I use this CNN Encoder --> Transformer Encoder --> Transformer Decoder frameeork in Bristol-Myers Squibb's Molecular Translation Kaggle competition.)*

#### Transformers in Object Detection

Attempting to use tranformers in object detectoin is a *significantly* more difficult task that in captioning or classification. The main problem is that state of the art detectors achieve success only by generating huge numbers of proposed object detections (often in the tens of thousands), and then weeding them down to single or double digit number of predictions with techniques such as Non-Max Suppression. Transformers simply take up too much memory to handle such a large numbers of object proposals. 

#### DETR's Object Detection Solution

Instead of producing thousands of bad predictions and whittling them down to a few good predictions, DETR directly predicts <= 100 high quality object detections using a novel take on the CNN Encoder --> Transformer Encoder --> Transformer Decoder framework. 

The decoder trains 100 detector objects, vectors whose role is to interact with the encoder through joint-attention and produce a single object prediction. Once trained, these detectors are fixed inputs to the Transformer Decoder, constants entirely independent of the image encodings they will interact with. This design as independent constants allows them to be trained in parallel without masking. 

DETR also replaces the non-trained techniques such as non-max suppression (NMS) with a fully trainable "end-to-end" process using bipartite matching. The result is an object detector that is the new state of the art for larger objects, but lags behind the top CNN-only models with smaller objects.


----
## Input Formats and Model Parameters

The model expects inputs values to be provided as dictionary of tensors. (Any valid TF format accepting string keys is acceptible). Images should be resized to uniform shape and targets should be padded to a uniform number of objects* (recommend using the max number in the training set.) The choice of image resizing / object padding does not matter, other than to be consistent within each run, but can vary across training runs or inference calls.

#### Mandatory Inference Key
- **'image'**: RGB image tensors scaled into [0,1]. shape = [batch, height, width, 3]
  
#### Mandatory Training Keys (in addition to inference keys)
- **'num_boxes'**: integer value of shape [batch, 1] indicating the number of target objects in the image
- **'bbox'**: floats (ymin, xmin, ymax, xmax) as scaled box coordinate values in [0,1] with shape [batch, padded_num_boxes, 4]. Any float value can be used for padding.
- **'category'**: strings of shape [batch, padded_num_boxes, 1]. Requires exactly one category per object. (Use '\<PAD\>' to indicate no category)
- **'attribute'**: strings of shape [batch, padded_num_boxes, num_attributes]. (Use '\<PAD\>' to indicate no attribute. In standard object detection (no attributes), tensors consist entirely of padding, shape = [batch, padded_num_boxes, 1].)

*Note: I found that using ragged tensors instead of padding to uniform number of objects slows down training by a factor of 3!*

### Model Paramaters

- **num_object_preds:** max number of predictions model is capable of outputing. The DETR paper suggests this will need to be much larger that the max number of objects in a picture. (2x as many in the examples they provided).
- **image_size:** size images will be resized to for the CNN encoder. (This need not be the size of input images)
- **num_encoder_blocks:** int >= 0. Number of encoder transformer blocks. Use 0 to skip to only use CNN encoder.
- **num_encoder_heads:** int >=1. Number of encoder transformer heads. 
- **encoder_dim:** int multiple of number of encoder heads >=1. 
- **num_decoder_blocks:** int >=1. Number of decoder transformer heads. 
- **num_decoder_heads:** int >=1. Number of decoder transformer heads. 
- **decoder_dim:** int multiple of number of decoder heads >=1. 
- **num_panoptic_heads:** (not yet implemented. Can pass None)
- **panoptic_dim:** (not yet implemented. Can pass 0)
- **vocab_dict:** Dictionary of form {'category': nonempty list of strings, 'attribute': potentially empty list of strings}. Do not include padding or mask values in these lists.

