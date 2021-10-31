# imports
import tensorflow as tf
import pandas as pd
import numpy as np


class Pipeline:
    """
    Inputs: dataframe of COCO-format annotations, including mandatory columns 
    ['image_path', 'width', 'height'] and optional columns 
    ['bbox', 'category', 'supercategory', 'attribute']. 

    Outputs: 
    - TF dataset object with labelled values ['image_id', 'image', 'bbox', 
        'category', 'supercategory', 'attribute'].
    - Labels are provided as ragged tensors to allow for variable numbers of objects per image. 
    - Bounding boxes are normalized as floats in [0,1] in same format as they arrived from dataframe
    - Image tensors can be either decoded into tensor or left serialized as jpg/png/gif
    - Image tensors have float values in [0,1] and are resized to uniform shape (user defined)
    - Image tensors can (optionally) have random augmentations dynamically applied 
    - If any of ['bbox', 'category', 'supercategory', 'attribute'] are missing from the 
        dataframe, dummy values are output for the missing info.
    """

    def __init__(self, image_size, **kwargs):

        self.target_size = image_size   
        self.TRANSLATION_TABLE = str.maketrans({'[':None, ']':None, ',':None, "'":None, '"':None, })

        # key labels
        self.image_path_key = 'image_path'
        self.image_key = 'image'
        self.bbox_key = 'bbox'
        self.category_key= 'category'
        self.attribute_key = 'attribute'
        self.num_objects_key = 'num_objects'
        self.image_id_key = 'image_id'

    # Helper functions
    # Key names
    def remove_keys(self, val):
        outputs = val[self.image_key], val[self.category_key], val[self.attribute_key], \
                    val[self.bbox_key], val[self.num_objects_key], val[self.image_id_key]
        return outputs

    def without_keys(self, ds):
        return ds.map(self.remove_keys)

    # ## image loaders
    def load_image(self, image_path):
        image_path = tf.squeeze(image_path)
        image = tf.keras.layers.Lambda(lambda x: tf.io.read_file(x))(image_path)
        return image   

    def decode_one_image(self, image):
        target_size=self.target_size

        image = tf.keras.layers.Lambda(lambda x: tf.io.decode_image(x, channels=3, expand_animations=False))(image)
        image = tf.keras.layers.experimental.preprocessing.Resizing(*target_size)(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image    
    
    def COCO_to_ymin_xmin_ymax_xmax(self, boxes):
        return np.concatenate([boxes[:, 1:2], 
                               boxes[:, 0:1],
                               boxes[:, 1:2] + boxes[:, 3:4],
                               boxes[:, 0:1] + boxes[:, 2:3]], axis=-1)

    def ymin_xmin_ymax_xmax_to_COCO(self, boxes):
        return np.concatenate([boxes[:, 0:1],
                               boxes[:, 1:2],    
                               boxes[:, 2:3] - boxes[:, 0:1],                            
                               boxes[:, 3:4] - boxes[:, 1:2]], axis=-1)

    def COCO_to_VOC(self, boxes):
        return np.concatenate([boxes[:, 0:1],
                               boxes[:, 1:2], 
                               boxes[:, 0:1] + boxes[:, 2:3],
                               boxes[:, 1:2] + boxes[:, 3:4]], axis=-1)

    def VOC_to_COCO(self, boxes):
        return np.concatenate([boxes[:, 0:1],
                               boxes[:, 1:2], 
                               boxes[:, 2:3] - boxes[:, 0:1],
                               boxes[:, 3:4] - boxes[:, 1:2]], axis=-1)

    
    # TF Dataset Creation
    def data_generator(self, labels_df, decode_images=True, stream_from_directory=False, 
                       box_pad_val=-10.0, string_pad_val='<PAD>'):
        
        """
        Converts image data into a Tensorflow Dataset. 
        Inputs: 
        parameters: instance of hyper_parameters.Parameters() class
        'stream_from_directory' argument has no effect when labels_df is passed.

        Training datasets creation:
        'labels_df' is required and must include columns 'image_path', and 'y_true'.
        
        Inference datasets creation:
        Pass either a label_df ('image_path' column required) or provide the 
        image directory path string in 'stream_from_directory=image_directory'
        """

        # get params
        target_size = self.target_size
        
        # Initial ds creation
        # Option 1: get info from dataframe
        if labels_df is not None:

            # shuffle
            df = labels_df.sample(frac=1)
            num_samples = len(df)

            # Create image path ds
            ds_image_path = tf.data.Dataset.from_tensor_slices(df['image_path'])
            
            # Create image id ds
            ds_image_id = tf.data.Dataset.from_tensor_slices(df['id_num'])
            
            # Create num_boxes ds
            if 'num_boxes' in df.columns: 
                max_num_obj = df['num_boxes'].max()
                num_boxes = df['num_boxes']
            else:
                max_num_obj = 1
                num_boxes = df['image_path'].apply(lambda x: 0)
            ds_num_objects = tf.data.Dataset.from_tensor_slices(num_boxes)
            
            # Create categories ds
            if 'category' in df.columns:
                category = df['category']
            else:
                category = df['image_path'].apply(lambda x: [[string_pad_val]])

            # create uniform shape tensors
            category = tf.ragged.constant(category.to_list())                
            category = category.to_tensor(shape=[num_samples, max_num_obj, 1], 
                                            default_value=string_pad_val)
                                            
            ds_category = tf.data.Dataset.from_tensor_slices(category)

            # Create attributes ds
            if 'attribute' in df.columns:
                attribute = df['attribute']
            else:
                attribute = df['image_path'].apply(lambda x: [[string_pad_val]])

            # create uniform shape tensors
            attribute = tf.ragged.constant(attribute.to_list())     
            max_attributes = attribute.bounding_shape()[-1]
            attribute = attribute.to_tensor(shape=[num_samples, max_num_obj, max_attributes], 
                                            default_value=string_pad_val)                                           
            ds_attribute = tf.data.Dataset.from_tensor_slices(attribute)


            # Create bbox ds
            if 'bbox' in df.columns:
                bbox = df['bbox']
            else:
                bbox = df['image_path'].apply(lambda x: [[box_pad_val, box_pad_val, box_pad_val, box_pad_val]])
            
            # ##create uniform shape tensors
            bbox = tf.ragged.constant(df['bbox'].to_list())                
            bbox = bbox.to_tensor(shape=[num_samples, max_num_obj, 4], 
                                            default_value=box_pad_val)
                                            
            ds_bbox = tf.data.Dataset.from_tensor_slices(bbox)

            # Merge into a single dataset
            ds = tf.data.Dataset.zip((ds_image_id, ds_image_path, ds_num_objects, ds_category, ds_attribute, ds_bbox))

            def apply_names(image_id, image_path, num_objects, category, attribute, bbox):
                return {'image_id':image_id, 'image_path':image_path, 
                        'num_objects':num_objects, 'category':category, 
                        'attribute':attribute, 'bbox':bbox}

            ds = ds.map(apply_names)
            
            # shuffle
            ds = ds.shuffle(50000)

            # decode images into array. (Note: uses tf.io, which requires unbatched images)
            def load(val):
                image_path = val[self.image_path_key]
                val[self.image_key] = self.load_image(image_path)      
                del val[self.image_path_key]          
                return val

            ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
            
            if decode_images:
                def decoder(val):
                    image = val[self.image_key]
                    val[self.image_key] = self.decode_one_image(image)
                    return val

                ds = ds.map(decoder, num_parallel_calls=tf.data.AUTOTUNE)
            
        # Option 2: get info from image file directory
        else:
            image_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=stream_from_directory, labels='inferred', label_mode=None,
                class_names=None, color_mode='rgb', batch_size=1, 
                image_size=target_size, shuffle=False, seed=None, validation_split=None, 
                subset=None, follow_links=False)
            
            # unbatch (needed for later ops)
            image_ds = image_ds.unbatch()

            # set filenames as image id and batch results
            image_id_ds = tf.data.Dataset.from_tensor_slices(image_ds.file_paths)
            image_id_ds = image_id_ds.map(lambda x: tf.strings.split(x, os.path.sep)[-1],
                                            num_parallel_calls=tf.data.AUTOTUNE)
            
            if decode_images is False:  
                # convert image to raw byte string. Note: cannot have batch dim for encoding
                image_ds = image_ds.unbatch()
                image_ds = image_ds.map(lambda x: tf.cast(x, dtype=tf.uint16))
                image_ds = image_ds.map(lambda image: tf.io.encode_png(image))
                image_ds = image_ds.map(lambda image: tf.io.serialize_tensor(image))
            
            # create  placeholder for (unknown) labels
            placeholder_num_obj_ds = image_id_ds.map(lambda x: -1, num_parallel_calls=tf.data.AUTOTUNE)
            placeholder_box_ds = image_id_ds.map(lambda x: [[box_pad_val, box_pad_val, box_pad_val, box_pad_val]], num_parallel_calls=tf.data.AUTOTUNE)
            placeholder_string_ds = image_id_ds.map(lambda x: [[string_pad_val]], num_parallel_calls=tf.data.AUTOTUNE)
                        
            # merge into single dataset
            ds = tf.data.Dataset.zip((image_ds, image_id_ds, placeholder_string_ds, 
                    placeholder_string_ds, placeholder_box_ds, placeholder_num_obj_ds))

            # name the elements for interpretability
            def map_names(image, image_id, a, b, c, d):
                return  {self.image_key: image, 
                         self.image_id_key: image_id, 
                         self.category_key: a, 
                         self.attribute_key: b, 
                         self.bbox_key: c, 
                         self.num_objects_key: d}
   
            ds = ds.map(map_names, num_parallel_calls=tf.data.AUTOTUNE)

        # display summary
        print(ds.element_spec, '\n')
        
        return ds


# Image Augmentations for Training Data
class Augmentations(Pipeline):
    def __init__(self):
        super().__init__(image_size=None)
        
        """  inherited values
        self.image_path_key = 'image_path'
        self.image_key = 'image'
        self.bbox_key = 'bbox'
        self.category_key= 'category'
        self.attribute_key = 'attribute'
        self.num_objects_key = 'num_objects'
        self.image_id_key = 'image_id'
        """

    def random_downsizer_with_pad(self, image, bbox):
        """ Randomly shrinks image (doesn't preserve aspect ratio), shifts it
         and pads back to original size with whitespace. Bounding box values 
         are adjusted accordingly. """
        
        # choose random downsizing factors between 1 and ~3
        # choice is heavily weighted towards 1 (i.e. no change in size)
        rand_val = tf.math.maximum(1.0, tf.random.truncated_normal(
                        shape=([2]), mean=0.5, stddev=0.7))
        
        # get shapes
        orig_shape = tf.shape(image)[-3:-1]
        orig_shape_float = tf.cast(orig_shape, dtype=rand_val.dtype)

        # downsize, shift and pad image
        # ## resize
        new_shape = orig_shape_float / rand_val
        new_shape = tf.cast(new_shape, dtype=orig_shape.dtype)
        image = tf.image.resize(image, size=new_shape)
        
        # ## random shift values
        offset_height = tf.random.uniform(minval=0, maxval= orig_shape[0] - new_shape[0] + 1,  
                                          shape=[], dtype=orig_shape.dtype)
        offset_width = tf.random.uniform(minval=0, maxval= orig_shape[1] - new_shape[1] + 1,  
                                         shape=[], dtype=orig_shape.dtype)

        # ## pad back to orig size with whitespace
        image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                            target_height=orig_shape[0], target_width=orig_shape[1])

        # downsize and shift bbox
        # ## downscale box
        denom =  [tf.concat([rand_val, rand_val], axis=-1)]
        bbox = bbox / denom
        
        # ## normalize shift values
        offset_height = tf.cast(offset_height / orig_shape[0], dtype=bbox.dtype)
        offset_width = tf.cast(offset_width / orig_shape[1], dtype=bbox.dtype)
        
        # ## shift boxes
        shift = tf.concat([[[offset_height]], [[offset_width]], [[offset_height]], [[offset_width]]], axis=-1)
        bbox = bbox + shift

        return image, bbox

    ## random contrast
    def random_contrast(self, image, lower=.8, upper=1.2):
        image = tf.image.random_contrast(image, lower=lower, upper=upper)
        return image

    ## random brightness
    def random_brightness(self, image, max_delta=.1):
        image = tf.image.random_brightness(image, max_delta=max_delta)
        return image

    ## random image quality
    def random_quality(self, image, min_quality=70, max_quality=100):
        image_shape = tf.shape(image)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality=min_quality, 
                                             max_jpeg_quality=max_quality)
        image = tf.reshape(image, [image_shape[0], image_shape[1], image_shape[2]])
        return image

    ## random saturation
    def random_saturation(self, image, min_saturation=.8, max_saturation=1.2):
        image = tf.image.random_saturation(image, lower=min_saturation, upper=max_saturation)
        return image


    # Combined Image Augmentation path
    def apply_image_augmentations(self, dataset):
        # NOTE: dataset elements arrive nbatched as dict with 'image' in keys

        # random_downsizer_with_pad. 
        def temp_mapper(val):
            image = val[self.image_key]
            bbox = val[self.bbox_key]
            val[self.image_key], val[self.bbox_key] = self.random_downsizer_with_pad(image, bbox)
            return val
        
        dataset = dataset.map(temp_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # random_contrast
        def temp_mapper(val):
            image = val[self.image_key]
            val[self.image_key] = self.random_contrast(image)
            return val
        dataset = dataset.map(temp_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        # random_brightness
        def temp_mapper(val):
            image = val[self.image_key]
            val[self.image_key] = self.random_brightness(image)
            return val
        dataset = dataset.map(temp_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        # random_quality
        def temp_mapper(val):
            image = val[self.image_key]
            val[self.image_key] = self.random_quality(image)
            return val
        dataset = dataset.map(temp_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        # random_saturation
        def temp_mapper(val):
            image = val[self.image_key]
            val[self.image_key] = self.random_saturation(image)
            return val
        dataset = dataset.map(temp_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

class TFRecordsConversions(Pipeline):
    def __init__(self, image_size, **kwargs):
        super().__init__(image_size)
        self.image_size = image_size

        """  inherited values
        self.image_path_key = 'image_path'
        self.image_key = 'image'
        self.bbox_key = 'bbox'
        self.category_key= 'category'
        self.attribute_key = 'attribute'
        self.num_objects_key = 'num_objects'
        self.image_id_key = 'image_id'
        """

    # TF Records Conversion
    # NOTE: Requires unbatched ds with raw images (i.e. images have not been decoded)
    def num_shards_needed(self, num_files, files_per_shard):
        num_shards = num_files // files_per_shard
        if  num_files % files_per_shard > 0:
            num_shards = num_shards + 1

        return num_shards
    
    # ## Conversion: Tensor - > Example
    def _bytes_feature(self, x):
        """Returns a bytes_list from a flat string / byte."""
        if isinstance(x, type(tf.constant(0))):
            x = tf.train.BytesList(value=[x.numpy()])
        return tf.train.Feature(bytes_list=x)

    def _float_feature(self, x):
        """ Returns a float_list from a flat float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[x]))

    def _int64_feature(self, x):
        """ Returns an int64_list from a flat bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

    def _non_flat_feature(self, x):
        """ Returns a bytes_list from a multidim tensor."""
        x = tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])
        return tf.train.Feature(bytes_list=x)

    # converts single ds element to TF Example
    def serialize_example(self, bbox, attribute, category, num_objects, image_id, image):

        feature = {self.bbox_key: self._non_flat_feature(bbox),
                   self.attribute_key: self._non_flat_feature(attribute),
                   self.category_key: self._non_flat_feature(category),
                   self.num_objects_key: self._int64_feature(num_objects),
                   self.image_id_key: self._int64_feature(image_id),
                   self.image_key: self._bytes_feature(image)
            }

        feature = tf.train.Features(feature=feature)
        example_proto = tf.train.Example(features=feature)

        return example_proto.SerializeToString()

    def remove_labels(self, val):
        bbox = val[self.bbox_key]
        attribute = val[self.attribute_key]
        category = val[self.category_key]
        num_objects = val[self.num_objects_key]
        image_id = val[self.image_id_key]
        image = val[self.image_key]

        return bbox, attribute, category, num_objects, image_id, image

    def tf_serialize_example(self, *x):        
        tf_string = tf.py_function(self.serialize_example, [*x], tf.string)
        return tf.reshape(tf_string, ()) # The result is a scalar.

    # converts tensor Dataset -> TF Record Files
    def write_TFRecords(self, ds, num_files, files_per_shard, name_qualifier=None):

        ds = ds.map(self.remove_labels)
        ds = ds.map(self.tf_serialize_example)

        elements_remaining = True
        file_number = 0

        while elements_remaining:
            
            # create dataset segment for next record
            try:  
                ds_i = ds.skip(file_number*files_per_shard).take(files_per_shard)
            except:
                elements_remaining = False
                break
            
            # write next record
            filename = f'{name_qualifier}_data_{file_number}.tfrecord'
            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(ds_i)

            file_number = file_number + 1

            if tf.math.greater_equal(file_number, num_files // files_per_shard):
                elements_remaining = False
                break

        print(f'Completed. {file_number} files written.')
        return None
    
    
    # ## Conversion: Example -> Tensor
    def decode_one_image(self, image):
        target_size = self.image_size

        image = tf.keras.layers.Lambda(lambda x: tf.io.decode_image(x, channels=3, 
                                                 expand_animations=False))(image)
        image = tf.keras.layers.experimental.preprocessing.Resizing(*target_size)(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image   

    def decode_one_float(self, x):
        x = tf.io.parse_tensor(x, out_type=tf.float32)
        return x

    def decode_one_string(self, x):
        x = tf.io.parse_tensor(x, out_type=tf.string)
        return x

    def parse_one_example(self, example):
        # convert to feature
        feature_description = {
            self.bbox_key: tf.io.FixedLenFeature([], tf.string, default_value=''),
            self.attribute_key: tf.io.FixedLenFeature([], tf.string, default_value=''),
            self.category_key: tf.io.FixedLenFeature([], tf.string, default_value=''),
            self.image_id_key: tf.io.FixedLenFeature([], tf.int64, default_value=0),
            self.num_objects_key: tf.io.FixedLenFeature([], tf.int64, default_value=0),
            self.image_key: tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        feature = tf.io.parse_single_example(example, feature_description)

        # convert bytestrings back to ragged tensors
        feature[self.bbox_key] = self.decode_one_float(feature[self.bbox_key])
        feature[self.attribute_key] = self.decode_one_string(feature['attribute'])
        feature[self.category_key] = self.decode_one_string(feature[self.category_key])
        feature[self.image_key] = self.decode_one_image(feature[self.image_key])

        return feature
