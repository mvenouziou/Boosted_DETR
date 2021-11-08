# DETR for Tensorflow

This is my implementation of the DETR object detector in Tensorflow. It has been coded from first principles as presented in the paper [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Although I did not make use of their repository, the [official PyTorch implementation](https://github.com/facebookresearch/detr/tree/master) deserves citation.

The description below outlines unique features, novel model architectures ideas, and a brief history of concepts leading to DETR.

---
**The Official DETR was trained on COCO for the equivalent of 1152 GPU hours!** I have nothing remotely close to processing to match that... but let's see what my modified DETR can achieve on Google Colab setup training on the smaller Fashionpedia (COCO-format) dataset. This model includes an additional prediction head to detect auxillary features in the dataset. (COCO is not annotated in this way.) The dataset also includes many small objects, which official DETR exhibited difficulties with.

- *@48 GPU Hours: a big jumble of many false positives. the model is still getting its bearings.*
- <img src="https://github.com/mvenouziou/DETR_for_TF/blob/main/validation_sample_image_day_%202_of_48.png" alt="Day 2" style="width:200px;"/>

- *@96 GPU Hours: The model has learned prior probabilities and is selective with predictions. It understands locations (ex. a belt belongs on the waist), but produces false positives.*
- <img src="https://github.com/mvenouziou/DETR_for_TF/blob/main/validation_sample_image_day_4_of_48.png" alt="Day 4" style="width:400px;"/>

- *@120 GPU Hours: Strong improvements in learned priors and selectivity in predictions. It is increasingly accurate on broad categories ("pants"), with intelligent (but incorrect) predictions of auxillary features that belong together with the main category ("pockets, belts").*
- <img src="https://github.com/mvenouziou/DETR_for_TF/blob/main/validation_sample_image_day_5_of_48.png" alt="Day 5" style="width:300px;"/>

----

## Model Features:

**Fully Integrated with Tensorflow 2 / Keras API**
  -   Alternate approach than the Tensorflow Object Detection API, which lives outside Tensorflow's normal build/train API.
  -   Straightforward modification, inference and training  for anyone versed in Tensorflow.  
  -   Train model in single and distributed GPU environments by simply passing an optimizer into *model.compile()* and then calling *model.fit()*.  Custom loss functions, metrics, and training regime have all been built in.

**Predict Fine-Grained Object Descriptions**
  -   Allows richer descriptions than traditional single-class object detection. (The data pipeline produces masked dummy features when corresponding training data is not provided.)

**Classifier Sub-Model for Pretraining**
  -   (Note: training benefits of this sub-model provides not yet evaluated/optimized.)  

**Built-in Text Tokenization / De-Tokenization**
  -   All model inputs and ouputs are human-readable, with no discernable cost to training or inference speed.
  -   Note that while text data isn't compatible with Tensroflow TPU training at this time, TPU incompatibility is already a consequence of a fundamental feature of the DETR architecture. (See notes section below.)

**Custom Data Pipeline**
  -   Automatically load COCO-format object detection data as TF Datasets with optional image augmentations.
  -   All class / subclass information is presented to the user as text. (Standard COCO datasets are typically pre-tokenized, requiring de-tokenization in order to interpret values.)

----
## Novel Architectures:
*The DETR paper authors used the equivalent of 1,152 GPU-hours of compute time (3 days on 16 GPU!), which prevents me from training the base model, let alone experiment with novel ideas. However, here are possible architecture experiments for people with access to better resources:*

**Adaptive Decoder Inference Size**

*Standard DETR trains a shared prediction head to produce outputs after each decoder block. During inference, however, intermediate predictions are not produced. (One reason for this training regime is so that a single large model can be trained and then cut down to desired size based on application.)*

*Alternative: Predict confidence estimates after each deder block. If a threshhold is reached, output predictions from that block and skip subsequent decoder blocks.*

  -   Pro: "Easy" images use fewer computations. "Difficult" images still have access to the full model.
  -   Pro: Allows larger decoder network with the same average image inference cost.
  -   Con: Added inference & training costs to create reliable confidence level predictions after every decoder block.
  -   Unkown: Net effect on inference quality.

**Adaptive Encoder + Decoder Size**

*Use an adaptive encoder as well as adaptive decoder size.*

  -   Pro: Adds variety to the number of encoder features seen by the decoder.
  -   Pro: encoder blocks deferred until needed. Imporved inference speed at any given decoder block, since fewer encoder blocks used.
  -   Con: Alters model architecture in a way likely to increase training time on what is already a computationally intensive training regime.
  -   Unkown: Effect on prediction quality and early stopping decisions

**Boosted Ensemble**

*Extension of the adaptive encoder/decoder architecture proposed above. Standard DETR was found to perform below state-of-the-art when dealing with small objects. The leading small-object detectors all have shared prediction heads that access features from multiple (convolutional) encoder scales, and produce a much larger number of predictions. The adaptive encoder/decoder DETR proposal is analogous to using CNN courseness levels to detect objects with a wide variety of sizes and increases the number of predictions made.*

-   Option 1: Instead of early-stopping, carry forward the highest confidence predictions from each decoder block, then perform an additional bipartite matching training step
-   Option 2 (Boosted ensemble): Retain high confidence predictions, and carry the corresponding decoder features forward at each step via attention masking. Decoder features corresponding to low confidence predictions get updated during the next decoder block.

----
## Notes:

- My implementation is in contrast to the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), which is not compatible with the standard Tensorflow model building/training API. Their approach add significant complexity to making more than small modifications to their prebuilt models. Perhaps their choice allows better portability to TF Lite and JS formats.

- Model training (but not inference) uses a SciPy function, *linear_sum_assignment*, for bipartite matching. Unfortunately this function is not compatible with Tensorflow TPU training and I have not yet found a workable alternative. (The official DETR model also relies on this function.)

- Panoptic segmentation not yet implemented.

----

##  Background

After Google and others showed great success in Natural Language Processing with Attention Transformers (developed in the paper [*Attention is All You Need*](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (2017)), researchers quickly began adapting Transformers to image processing tasks. [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) (2020) a.k.a. DETR is a novel extension of Attention Transformers to Object Detection.

#### Transformers in NLP

RNN's with Attention had become the defacto standard for NLP tasks. These, however, are hindered by lack of training parallelism, slowdowns on large dimensions, and reduced quality on long sequences. *Attention is All You Need* replaced RNN's entirely with a modified attention architecture ("Transformers") utilizing joint encoder-decoder attention and decoder self-attention. Their implementation enhances the benefits of attention (allowing the decoder to access to the full set of encoder vectors and known decoder vectors) but gains efficiencies in parallelized training, and linear projections onto low-dimensional subspaces,

#### Transformers in Image Captioning

State of the Art Image processing using Convolutional Neural Networks do not suffer from NLP's previous lack of parallelism, removing one of the main benefits of switching to transformers. In addition, transformers have an enormous memory footprint compared to CNNs, and they lack CNNs natural sense of spacial positioning.

The most likely benefit of transformers would be for tasks involving both image processing and NLP, such as imaging captioning. [*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*](https://proceedings.mlr.press/v37/xuc15.pdf) (2015) had already used a hybrid approach connecting a CNN encoder with an RNN decoder. Replacing the RNN with Transformers was a natural extension.

*(See my project [Attention is What You Get](https://github.com/mvenouziou/Project-Attention-Is-What-You-Get) where I use the CNN Encoder --> Transformer Encoder --> Transformer Decoder frameeork in Bristol-Myers Squibb's Molecular Translation Kaggle competition.)*

#### Transformers in Object Detection

Attempting to use tranformers in object detectoin is a *significantly* more difficult task than in captioning or classification. The main problem is that state of the art detectors achieve success only by generating huge numbers of proposed object detections (often in the thousands or tens of thousands), and then weeding them down to single or double digit number of predictions with techniques such as Non-Max Suppression. Transformers simply take up too much memory to handle such a large numbers of object proposals.

#### DETR's Object Detection Solution

Instead of producing thousands of bad predictions and whittling them down to a few good predictions, DETR directly predicts a small number of high quality object detections using a novel take on the CNN Encoder --> Transformer Encoder --> Transformer Decoder framework.

The decoder trains individual detector objects, vectors whose role is to interact with the encoder through joint and self-attention. Each detector is responsible for a single object prediction. Once trained, these detectors are treated as fixed inputs to the Transformer Decoder, constants entirely independent of the image encodings they will interact with. This design as independent constants allows them to be trained in parallel without masking.

DETR also replaces the non-trained techniques such as non-max suppression (NMS) with a fully trainable "end-to-end" process using bipartite matching. The resulting object detection model became the new state of the art for larger object detection, but lags behind in small-object detection.


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
