# imports
import zipfile
from zipfile import ZipFile
import multiprocessing
import os
import sys
import shutil 
import json
import requests
import wget
import glob
import pandas as pd
import numpy as np



# Helper Class to load standard COCO-format detection + segmentation data
# Base Class for all datasets
# Subclasses for specific data sources follow
class GetDataset:
    """
    1. Downloads external compressed datasets to local zip directory.
    2. Extracts files from compressed files to local directory.

    Note: default setting extracts files to a subdir of '/content/datasets',
    which is a temporary file location when using Google Colab.
    Make sure to manually enter a save directory if this is not intended behavior.
      
    # Sample usage with Google Colab and Google Drive:

    # ## Set filepaths:
    archive_base_dir = '/content/drive/MyDrive/ShuffleExchange/datasets'  # Google Drive (long term storage).
    local_base_dir = '/content/datasets'  # Google Colab working directory. (Automatically deleted by Colab upon session exit.)

    # ## Download Data:
    data_loader = Fashionpedia(archive_base_dir=archive_base_dir, local_base_dir=local_base_dir)
    data_loader.get_data(download=True)  # downloads from remote source to archive directory
    data_loader.get_data(unzip=True)  # extracts from archive to working directory

    # ## Load Dataset Info
    df = data_loader.dataframes(subset='valid')  # returns annotations dataframe with mandatory keys 'annotations' and 'image_ path'
    data_loader.citation()  # prints dataset metadata
    """

    def __init__(self, dataset_name, archive_base_dir, local_base_dir,
                 image_source_filepath_dict, annotation_source_filepath_dict): 

        self._dataset_name = dataset_name
        self._archive_base_dir = archive_base_dir
        self._local_base_dir = local_base_dir
        self._image_source_filepath_dict = image_source_filepath_dict
        self._annotation_source_filepath_dict = annotation_source_filepath_dict
        self._pad_value = '<PAD>'

        # set save locations
        self._archive_image_dir = os.path.join(self._archive_base_dir, self._dataset_name, 'images')
        self._archive_annotation_dir = os.path.join(self._archive_base_dir, self._dataset_name, 'annotations')
        self._image_dir = os.path.join(self._local_base_dir, self._dataset_name, 'images')
        self._annotations_dir = os.path.join(self._local_base_dir, self._dataset_name, 'annotations')

    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def filepath_exist(self, filepath, force_rebuild):
        # check if file was already extracted
        already_exists = os.path.exists(filepath)
        
        if already_exists and not force_rebuild: 
            print(filepath, 'found. Using previously extracted data. (Note: set force_rebuild=True to override)')
            return True
        else:
            return False

    def download_to_archive(self, source_dir, source_filename, subset, datatype, qualifier=None, force_rebuild=False):

        if datatype == 'image':
            destination_dir = os.path.join(self._archive_image_dir, subset)

        elif datatype == 'annotation':
            destination_dir = os.path.join(self._archive_annotation_dir, subset)
        
        destination_filepath = os.path.join(destination_dir, source_filename)
        source_filepath = os.path.join(source_dir, source_filename)

        # check if file was already downloaded
        exists = self.filepath_exist(destination_filepath, force_rebuild)
        
        if not exists:
            # create destination directory structure
            self.create_folder(destination_dir)

            # overwrite any existing file
            if os.path.exists(destination_filepath):
                os.remove(destination_filepath)
            wget.download(source_filepath, destination_filepath)
                
            downloaded = True

        else:
            downloaded = False
        
        return downloaded

    def download_to_local_dir(self, source_file, datatype, subset, force_rebuild):

        if datatype == 'image':
            destination_dir = os.path.join(self._image_dir, subset)
        elif datatype == 'annotation':
            destination_dir = os.path.join(self._annotations_dir, subset)
        else:
            print('must set datatype = image or annotation')     

        filename = os.path.split(source_file)[1]
        destination_path = os.path.join(destination_dir, filename)
        # check if file was already downloaded
        exists = self.filepath_exist(destination_path, force_rebuild)
        if exists:
            downloaded = False

        else: # create local directory and copy file
            self.create_folder(destination_dir)
            shutil.copyfile(source_file, destination_path)
            downloaded = True           

        return downloaded

    def unzip_to_local_dir(self, source_file, datatype, subset, force_rebuild):

        if datatype == 'image':
            destination_dir = os.path.join(self._image_dir, subset)

        elif datatype == 'annotation':
            destination_dir = os.path.join(self._annotations_dir, subset)

        exists = self.filepath_exist(destination_dir, force_rebuild)
        if exists:
            extracted = False

        else:
            # create local directory
            if not os.path.exists(destination_dir):
                self.create_folder(destination_dir)

            # extract file and return to working directory
            cwd = os.getcwd()
            
            with zipfile.ZipFile(source_file, 'r') as zipped:         
                zipped.extractall(destination_dir)
            extracted = True             

        return extracted

    def get_data(self, download=False, unzip=False, force_rebuild=False):
        """ downloads to archive and extracts to local dir """

        base_dir = self._image_source_filepath_dict['base_dir']

        if download:
            for subset in ['train', 'test']:

                # images
                base_dir = self._image_source_filepath_dict['base_dir']
                qualifier = self._image_source_filepath_dict['qualifier']

                for filename in self._image_source_filepath_dict[subset]:

                    print('downloading:', filename)
                    self.download_to_archive(base_dir, filename, subset, datatype='image', 
                                            qualifier=qualifier, force_rebuild=force_rebuild)

                # annotations
                qualifier = self._image_source_filepath_dict['qualifier']
                base_dir = self._annotation_source_filepath_dict['base_dir']

                for filename in self._annotation_source_filepath_dict[subset]:
                    print('downloading:', filename)
                    self.download_to_archive(base_dir, filename, subset, datatype='annotation', 
                                            qualifier=qualifier, force_rebuild=force_rebuild)
        
        if unzip:                       
            for subset in ['train', 'test']:
                
                # get images
                base_image_path = os.path.join(self._archive_image_dir, subset)

                if os.path.exists(base_image_path):
                    image_filepath_list = os.listdir(base_image_path)
                    
                    for source_file in list(reversed(image_filepath_list)):
                        print('extracting:', source_file)

                        datatype = 'image'
                        source_filepath = os.path.join(base_image_path, source_file)
                        
                        if zipfile.is_zipfile(source_filepath):
                            self.unzip_to_local_dir(source_filepath, datatype, subset, force_rebuild)
                        else:
                            self.download_to_local_dir(source_filepath, datatype, subset, force_rebuild)

                # get annotations
                base_annotation_path = os.path.join(self._archive_annotation_dir, subset)
                if os.path.exists(base_annotation_path):
                    annotation_filepath_list = os.listdir(base_annotation_path)
        
                    for source_file in annotation_filepath_list:
                        datatype = 'annotation'
                        source_filepath = os.path.join(base_annotation_path, source_file)

                        # copy / extract
                        if zipfile.is_zipfile(source_filepath):
                            self.unzip_to_local_dir(source_filepath, datatype, subset, force_rebuild)
                        else:
                            self.download_to_local_dir(source_filepath, datatype, subset, force_rebuild)
        
        return None       


class COCOStandard(GetDataset):

    def __init__(self, local_base_dir=None, archive_base_dir=None, **kwargs):

        dataset_name = 'COCO'

        # Location of Source Files
        external_base_dir = 'http://images.cocodataset.org'

        image_source_filepath_dict = {
            'base_dir': os.path.join(external_base_dir, 'zips'),
            'qualifier': {'name': None,},
            'train': ['train2017.zip'],
            'test': ['val2017.zip']#['test2017.zip', 'val2017.zip']
        }

        annotation_source_filepath_dict = {
            'base_dir': os.path.join(external_base_dir, 'annotations'),
            'qualifier': {'name': None,},           
            'train': ['stuff_annotations_trainval2017.zip',
                      'panoptic_annotations_trainval2017.zip',
                      'annotations_trainval2017.zip'],  
            'test': []}  # train list includes both validation data

        # initialize class
        super().__init__(dataset_name, archive_base_dir, local_base_dir,
                          image_source_filepath_dict, annotation_source_filepath_dict)

        # placeholder attributes (filled by later function calls)
        self._prepared_info = None
        self._category_indx_to_name = None
        self._attribute_indx_to_name = None

    def prepare_COCO_from_json(self, subset, force_rebuild=False):

        if subset == 'train':
            image_dir = '/content/COCO/images/train/train2017'
            json_filenames = ['/content/COCO/annotations/train/annotations/instances_train2017.json',
                              '/content/COCO/annotations/train/annotations/person_keypoints_train2017.json',
                              '/content/COCO/annotations/train/annotations/captions_train2017.json'
                              ]

        elif subset == 'val':
            image_dir = '/content/COCO/images/test/val2017'
            json_filenames = ['/content/COCO/annotations/train/annotations/instances_val2017.json',
                              '/content/COCO/annotations/train/annotations/person_keypoints_val2017.json',
                              '/content/COCO/annotations/train/annotations/captions_val2017.json'
                             ]

        instance_info = self.load_COCO_json(json_filenames[0])
        #keypoint_info = self.load_COCO_json(json_filenames[1]) # not yet coded to handle multiple sets
        #caption_info = self.load_COCO_json(json_filenames[2]) ####################### not yet appropriately coded for this file

        # create or load annotations df
        # check for previously prepared ds
        possible_filepath = (os.path.join(self._archive_annotation_dir, f'{subset}_combined_annotations_df.json'))
        
        if os.path.exists(possible_filepath) and not force_rebuild:
            combined_annotations_df = pd.read_json(possible_filepath)

        else:  # create df from scratch
            combined_annotations_df = self.create_combined_COCO_detections_df(
                        instance_info['images_df'], instance_info['annotations_df'])

            combined_annotations_df['image_path'] = \
                combined_annotations_df['file_name'].apply(lambda x: os.path.join(image_dir, x))

            # update index to reclaim 'image_id' column
            combined_annotations_df = combined_annotations_df.reset_index().rename(columns={'image_id':'id_num'})

            # save to json
            combined_annotations_df.to_json(os.path.join(self._archive_annotation_dir, 
                                                     f'{subset}_combined_annotations_df.json'))
        
        self._prepared_info = {'annotations_df': combined_annotations_df, 
                               'categories_df': instance_info['categories_df'], 
                               'meta_info': instance_info['meta_info']}   

        # Display info
        print('Returns dictionary with keys:', self._prepared_info.keys())
        print()
        print('annotations_df:', self._prepared_info['annotations_df'].columns)
        print()
        print('note: bbox provided in normalized COCO format: [xmin, ymin, width, height]')

        return self._prepared_info


    def load_COCO_json(self, filename):

        # load json annotation files
        with open(filename) as f:
            all_info = json.load(f)

        meta_info = all_info['info']

        # images
        images_df = pd.DataFrame(all_info['images'])
        images_df = images_df.set_index('id')

        # annotations
        annotations_df = pd.DataFrame(all_info['annotations'])
        annotations_df = annotations_df.set_index('image_id')

        # category names
        if 'categories' in all_info.keys():
            categories_df = pd.DataFrame(all_info['categories'])
            self._category_indx_to_name = categories_df.set_index('id')['name'].to_dict()
        else:
            categories_df = None

        # attribute names (Fashionpedia dataset)
        if 'attributes' in all_info.keys():
            attribute_df = pd.DataFrame(all_info['attributes'])
            self._attribute_indx_to_name = attribute_df.set_index('id')['name'].to_dict()
        else:
            categories_df = None

        return {'categories_df': categories_df, 'annotations_df': annotations_df, 
                'images_df': images_df, 'meta_info': meta_info}

    def create_combined_COCO_detections_df(self, images_df, annotations_df):
        
        # prepare annotations dictionary
        # begin with columns included in all sets' images_df
        combined_annotations_dict = {'image_id': [], 'width':[], 'height':[], 
                                     'coco_url':[], 'file_name':[],}
        
        # object count
        if 'bbox' in annotations_df.columns:  # not used in captions
            combined_annotations_dict['num_boxes'] = []
            combined_annotations_dict['bbox'] = []

        # use strings instead of integer category labels
        if 'category_id' in annotations_df.columns:  # not used in captions
            def cat_name(cat_id):
                return self._category_indx_to_name[cat_id]

            combined_annotations_dict['category'] = []
            annotations_df['category_id'] = annotations_df['category_id'].apply(cat_name)

        # use strings instead of integer attribute labels
        if 'attribute_ids' in annotations_df.columns:  # used in Fashionpedia
            def att_name(att_id_list):
                return [self._attribute_indx_to_name[att_id] for att_id in att_id_list]

            combined_annotations_dict['attribute'] = []            
            annotations_df['attribute'] = annotations_df['attribute'].apply(att_name)

        for keyname in annotations_df.columns:  # handles other common formats such as keypoints
            if keyname in ['id', 'bbox', 'category_id', 'attribute_ids']:
                continue
            combined_annotations_dict[keyname] = []

        # function to consolidate object values from individual column
        def gather_objects(keyname, location, num_boxes):
            if num_boxes > 1:
                combined_info = []
                for info in annotations_df.loc[location][keyname]:
                    combined_info.append([info])                        
            else: 
                combined_info = [[annotations_df.loc[location][keyname]]]
            return combined_info

        # function to consolidate all object values for single image
        def combine_annotation_info(location):

            # image ID
            combined_annotations_dict['image_id'].append(location)
           
            # image dims
            width = images_df.loc[location]['width']
            height = images_df.loc[location]['height']
            combined_annotations_dict['width'].append(width)
            combined_annotations_dict['height'].append(height)

            # file name
            file_name = images_df.loc[location]['file_name']
            combined_annotations_dict['file_name'].append(file_name)

            # url
            coco_url = images_df.loc[location]['coco_url']
            combined_annotations_dict['coco_url'].append(coco_url)

            # Object Detection
            # bboxes & number of bboxes
            if 'bbox' in annotations_df.columns:

                # ## record number of boxes
                num_boxes = 0
                combined_bbox = []
                
                if type(annotations_df.loc[location]['bbox']) is list:
                    combined_bbox = annotations_df.loc[location]['bbox']
                    num_boxes = 1
                else:
                    for bbox in annotations_df.loc[location]['bbox']:
                        combined_bbox.append(bbox)
                        num_boxes += 1               

                combined_annotations_dict['num_boxes'].append(num_boxes)
                
                # ## add normalized bbox
                combined_bbox = np.array(combined_bbox) / np.array([[width, height, width, height]])
                combined_bbox = combined_bbox.tolist()
                combined_annotations_dict['bbox'].append(combined_bbox)
                        
            # category
            if 'category_id' in annotations_df.columns:
                combined_category_id = gather_objects('category_id', location, num_boxes)
                combined_annotations_dict['category'].append(combined_category_id)

            # area
            if 'area' in annotations_df.columns:
                combined_area = gather_objects('area', location, num_boxes)
                combined_annotations_dict['area'].append(combined_area)

            # iscrowd
            if 'iscrowd' in annotations_df.columns:
                if num_boxes > 1:
                    iscrowd = annotations_df.loc[location]['iscrowd'].iloc[0]  # only one unique value per image
                else:
                    iscrowd = annotations_df.loc[location]['iscrowd']
                combined_annotations_dict['iscrowd'].append(iscrowd)

            # Optional Components
            # Segmentation
            if 'segmentation' in annotations_df.columns:

                if type(annotations_df.loc[location]['segmentation']) is list:
                    combined_segmentation = [annotations_df.loc[location]['segmentation']]
                
                else:
                    combined_segmentation = []
                    for segmentation in annotations_df.loc[location]['segmentation']:
                        combined_segmentation.append(segmentation)

                combined_annotations_dict['segmentation'].append(combined_segmentation)

            if 'segments_info' in annotations_df.columns:
                combined_segments_info = gather_objects('segments_info', location, num_boxes)
                combined_annotations_dict['segments_info'].append(combined_segments_info)

            # Keypoints   
            if 'num_keypoints' in annotations_df.columns:
                num_keypoints = annotations_df.loc[location]['num_keypoints']
                combined_annotations_dict['num_keypoints'].append(num_keypoints)

            if 'keypoints' in annotations_df.columns:
            
                if type(annotations_df.loc[location]['keypoints']) is list:
                    combined_keypoints = [annotations_df.loc[location]['keypoints']]
                
                else:
                    combined_keypoints = []
                    for keypoints in annotations_df.loc[location]['keypoints']:
                        combined_keypoints.append(keypoints)

                combined_annotations_dict['keypoints'].append(combined_keypoints)

            # Captioning
            if 'caption' in annotations_df.columns:
                caption = annotations_df.loc[location]['caption']               
                combined_annotations_dict['caption'].append(caption)

            # DensePose
            for keyname in ['dp_I', 'dp_U', 'dp_V', 'dp_x', 'dp_y', 'dp_masks']:
                if keyname in annotations_df.columns:
                    
                    combined_info = []
                    
                    for keyname_info in annotations_df.loc[location][keyname]:
                        combined_info.append(keyname_info)
                        
                    combined_annotations_dict[keyname].append(combined_info)

            # Other Annotations (not in standard COCO sets)
            if 'attribute_ids' in annotations_df.columns:
                combined_info = []

                for attribute_info in annotations_df.loc[location]['attribute_ids']:
                    combined_info.append(attribute_info)
                    
                combined_annotations_dict['attribute'].append(combined_info)

            return None

        # Add each prepared image info to dict
        all_image_ids = list(annotations_df.index.unique())

        for image_id in all_image_ids:
            combine_annotation_info(image_id)

        # create dataframe indexed on 'image_id' (index for join)
        combined_annotations_df = pd.DataFrame(combined_annotations_dict)
        combined_annotations_df = combined_annotations_df.set_index('image_id')
        
        return combined_annotations_df

    # helper functions
    def max_num_obj(self):
        return self._prepared_info['annotations_df']['bbox'].apply(lambda x: len(x)).max()

    def dataframes(self):
        return self._prepared_info['annotations_df']

    def get_vocab(self):
        
        vocab = {'category': list(self._category_indx_to_name.values())}

        if self._attribute_indx_to_name:           
            vocab['attribute'] = list(self._attribute_indx_to_name.values())
        
        return vocab


class Fashionpedia(GetDataset):
    """
    Loads Fashionpedia Dataset, https://fashionpedia.github.io/home/index.html.
    """
    
    def __init__(self, local_base_dir=None, archive_base_dir=None, **kwargs):

        self._dataset_name = 'Fashionpedia'
        self._pad_value = '<PAD>'

        # Location of Source Files
        external_base_dir = 'https://s3.amazonaws.com/ifashionist-dataset/'

        image_source_filepath_dict = {
            'base_dir': os.path.join(external_base_dir, 'images'),
           
            'qualifier': {'name': None,},
            'train': ['train2020.zip'],
            'test': ['val_test2020.zip']
        }

        annotation_source_filepath_dict = {
            'base_dir': os.path.join(external_base_dir, 'annotations'),
            'qualifier': {'name': None,},           
            'train': ['instances_attributes_train2020.json'],
                      #'attributes_train2020.json'],
            'test': ['instances_attributes_val2020.json']
                     #'attributes_val2020.json']
        }

        # initialize class
        super().__init__(self._dataset_name, archive_base_dir, local_base_dir,
                          image_source_filepath_dict, annotation_source_filepath_dict)

        # placeholder attributes
        self._id_to_category = None 
        self._id_to_attribute = None
        self._attribute_vocab = None
        self._category_vocab = None

    # CITATION
    def citation(self):
        print('''Dataset provided with Creative Commons Attribution 4.0 License and is publicaly available at https://arxiv.org/pdf/1905.12794.pdf. 
        CITATION: @inproceedings{jia2020fashionpedia,
            title={Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset},
            author={Jia, Menglin and Shi, Mengyun and Sirotenko, Mikhail and Cui, Yin and Cardie, Claire and Hariharan, Bharath and Adam, Hartwig and Belongie, Serge}
            booktitle={European Conference on Computer Vision (ECCV)},
            year={2020}
            }''')

    # NOTES
    def notes(self):
        bbox_note = 'bbox provided in normalized COCO format: [xmin, ymin, width, height]'
        segmentation_note = 'segmentations not normalized and contain mix of polyhons and rle formats'
        print(bbox_note, '\n',segmentation_note)

        return 'normalized COCO'

    # json parsing function
    def json_to_dataframe(self, filename, save_subset):

        # load json annotation files
        with open(filename) as f:
            all_info = json.load(f)

        print(all_info.keys())
        # get index --> category maps
        category_df = pd.DataFrame(all_info['categories'])
        category_df = category_df.set_index('id').rename(columns={'name':'category'})
        id_to_category = category_df['category'].to_dict()

        # get attribute <--> index maps
        attribute_df = pd.DataFrame(all_info['attributes'])
        attribute_df = attribute_df.set_index('id').rename(columns={'name':'attribute'})
        id_to_attribute = attribute_df['attribute'].to_dict()
        
        dummy_indx = -1  # add NONE placeholder value
        id_to_attribute[dummy_indx] = self._pad_value
        
        # get box and class annotations
        df_image = pd.DataFrame(all_info['images']).set_index('id')
        df_annotations = pd.DataFrame(all_info['annotations']).set_index('image_id')

        # ## update category_id list nesting to match attribute_ids
        df_annotations['category_id'] = df_annotations['category_id'].apply(lambda x: [x])

        # Collect each image's annotations as a dictionary of lists
        images_dict = {}
        all_ids = df_image.index
        
        for id_num in all_ids:

            # restrict data to the specific image
            image_info_i = df_image.loc[id_num]
            annotation_info_i = df_annotations.loc[id_num]
            
            # get image meta info
            file_name = image_info_i['file_name']
            image_width = image_info_i['width']
            image_height = image_info_i['height']
            num_boxes = len(annotation_info_i)

            # collect box and annotation info.
            try:
                annotation_info_i.columns  # verifies this is a dataframe
                category_list = annotation_info_i['category_id'].to_list()
                attribute_list = annotation_info_i['attribute_ids'].to_list()
                segmentation_list = annotation_info_i['segmentation'].to_list()
                box_vals = np.array(annotation_info_i['bbox'].to_list())
                        
            except:  # entry is a series, there is only one object
                num_boxes = 1
                category_list = [annotation_info_i['category_id']]
                attribute_list = [annotation_info_i['attribute_ids']]
                segmentation_list = [annotation_info_i['segmentation']]
                box_vals = np.array([annotation_info_i['bbox']])        

            # replace category index with strings
            category_list = [[id_to_category[indx] 
                               for indx in cat] for cat in category_list]

            # pad missing attributes with <NONE> class and replace index with strings
            for i in range(num_boxes):
                if attribute_list[i] == []:
                    attribute_list[i] = [dummy_indx]
            attribute_list = [[id_to_attribute[indx] 
                               for indx in att] for att in attribute_list]

            # normalize box values into [0,1]
            box_vals = box_vals / np.array([image_width, image_height, image_width, image_height])
            box_vals = box_vals.tolist()
                    
            # save image results
            images_dict[id_num] = {'file_name': file_name,
                                   'bbox': box_vals,
                                   'category': category_list,
                                   'attribute': attribute_list,
                                   'segmentation': segmentation_list,
                                   'image_width': image_width, 
                                   'image_height': image_height,
                                   'num_boxes': num_boxes}

        # combine results and reclaim index column
        full_annotations_df = pd.DataFrame.from_dict(images_dict, orient='index')
        full_annotations_df = full_annotations_df.reset_index().rename(columns={'index':'id_num'})
        
        # save to json
        save_to_archive_path = os.path.join(self._archive_annotation_dir, save_subset, save_subset + '_full_annotations_df.json')
        full_annotations_df.to_json(save_to_archive_path)

        return full_annotations_df


    def dataframes(self, subset='train', force_rebuild=False):

        if subset not in ['train', 'val']:
            print('Error: subset must be either "train" or "val"')
            return None

        # select annotations file
        if subset=='train':
            save_subset = 'train'
        elif subset == 'val':
            save_subset = 'test'  # rename for updating image paths
        
        base_annotation_path = os.path.join(self._annotations_dir, save_subset)
        annotation_file_list = self._annotation_source_filepath_dict[save_subset]
        
        for filename in annotation_file_list:
            json_file = os.path.join(base_annotation_path, filename)

        save_to_archive_path = os.path.join(self._archive_annotation_dir, save_subset, save_subset + '_full_annotations_df.json')
        
        # check dataset was previously prepared
        already_exists = self.filepath_exist(save_to_archive_path, force_rebuild)

        if already_exists:
            print('Previously prepared datafile found. Loading from saved file.')
            print('Use "force_rebuild=True" to generate dataframe from scratch')
            full_annotations_df = pd.read_json(save_to_archive_path)
            
        else:
            full_annotations_df = self.json_to_dataframe(json_file, save_subset)

        # add image path info
        directory = os.path.join(self._image_dir, save_subset)
        full_annotations_df['image_path'] = full_annotations_df['file_name'].apply(lambda x: os.path.join(directory, save_subset, x))

        # save as attribute
        self._full_annotations_df = full_annotations_df

        return full_annotations_df

    # helper functions
    def max_num_obj(self):
        return self._full_annotations_df['bbox'].apply(lambda x: len(x)).max()

    def get_vocab(self):

        category_vocab = {elem for cat_list in self._full_annotations_df['category'].to_list()
                                for cat in cat_list for elem in cat}

        attribute_vocab = {elem for atts_list in self._full_annotations_df['attribute'].to_list() 
                                 for att in atts_list for elem in att}

        vocab = {'category': list(category_vocab),
                 'attribute': list(attribute_vocab)
                 }
        return vocab


class UnsplashLite(GetDataset):
    """
    Note: this utilizes Pandas' API, so follows their file name/structure conventions.

    Loads Unsplash.com's public use "lite" dataset containing 25,000 images
    with metadata including keywords and colors.
    Call self.citation() to print info.  As of writing, this dataset was
    made available for commercial usage subject to the terms at
    https://github.com/unsplash/datasets/blob/master/TERMS.md

    Sample usage:
    data_loader = UnsplashLite()
    data_loader.get_data(download=False, unzip=False)  # get files
    unsplash_df_dict = data_loader.dataframes()  # create dataframes
    unsplash_df_dict['keywords']['keyword'].unique()  # get all keywords
    """

    def __init__(self, local_base_dir=None, archive_base_dir=None, **kwargs):

        self._dataset_name = 'UnsplashLite'

        # Location of Source Files
        external_base_dir = 'https://unsplash.com/data/lite/'

        image_source_filepath_dict = {
            'base_dir': external_base_dir,
            'qualifier': {'name': None,},
            'train': ['latest'],
            'test': None
        }

        annotation_source_filepath_dict = {
            'base_dir': os.path.join(external_base_dir, 'annotations'),
            'qualifier': {'name': None,},           
            'train': None,
            'test': None,
        }

        # initialize class
        super().__init__(self._dataset_name, archive_base_dir, local_base_dir,
                          image_source_filepath_dict, annotation_source_filepath_dict)

    # CITATION
    def citation(self):
        print('''CITATION: Unsplash Lite Dataset 1.2.0, unsplash.com/data.
        Lite dataset: available for commercial and noncommercial usage, containing 25k nature-themed Unsplash photos, 25k keywords, and 1M searches.
        See https://github.com/unsplash/datasets for info and
        https://github.com/unsplash/datasets/blob/master/TERMS.md for usage rights''')


    def dataframes(self):
        """ This code is from the Unsplash.com API.
        Return dictionary containing pandas dataframes with names of the form:
        datasets('photos'), datasets('keywords'), datasets('collections'),
        datasets('conversions'), datasets('colors').
        See DOCS.md in the unzip dir for details.
         """

        # go to root dir
        #os.chdir('/')
        #save_dir = self.local_zip_extract_dir_dict['clothing_dataset_grigorev']
        #csv_file = os.path.join(save_dir, 'images.csv')

        documents = ['photos', 'keywords', 'collections', 'conversions', 'colors']
        datasets = {}
        
        for doc in documents:
            files = glob.glob(os.path.join(self._archive_image_dir, doc) + ".tsv*")

            subsets = []
            for filename in files:
                df = pd.read_csv(filename, sep='\t', header=0)
                subsets.append(df)

            datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

        print(f'This returns a dictionary of dataframes. \
        Dict Keys: {documents}')

        return datasets


class ClothingDatasetGrigorev(GetDataset):
    """
    Loads Alexey Grigorev's Clothing Dataset, from Github repository. 
    Dataset was made available for commercial purposes at time of this writing. 
    See https://github.com/alexeygrigorev/clothing-dataset#readme for acknowledgements.
   
    Sample usage:
    data_loader = ClothingDatasetGrigorev()
    data_loader.get_data(download=False, unzip=False)  # get files
    df_dict = data_loader.dataframes()  # create dataframes
    """

    def __init__(self, local_base_dir=None, archive_base_dir=None, **kwargs):

        name = 'ClothingDatasetGrigorev'
        external_source_dict = {'clothing_dataset_grigorev': 'https://github.com/alexeygrigorev/clothing-dataset.git'}
        archive_save_dir_dict = {'clothing_dataset_grigorev': None}  # get_data() overrident for Github download
        
        local_zip_extract_dir_dict = {'clothing_dataset_grigorev': name}
        local_copy_dir_dict = {'clothing_dataset_grigorev': name}
        
        # initialize class
        super().__init__(local_base_dir=local_base_dir,
                         archive_base_dir=archive_base_dir,
                         external_source_dict=external_source_dict,
                         archive_save_dir_dict=archive_save_dir_dict,
                         local_zip_extract_dir_dict=local_zip_extract_dir_dict,
                         local_copy_dir_dict=local_copy_dir_dict,
                         name=name)

    # CITATION
    def citation(self):
        print('''CITATION: Alexey Grigorev's Clothing Dataset
            Github: https://github.com/alexeygrigorev/clothing-dataset
            Medium Post: https://medium.com/data-science-insider/clothing-dataset-5b72cd7c3f1f
            Full resolution photos: https://www.kaggle.com/agrigorev/clothing-dataset-full/''')

    def get_data(self, **kwargs):
        """ Gets dataset by cloning Github repo. Overrides base class def. """
        
        source = self.external_source_dict['clothing_dataset_grigorev']
        destination = self.local_zip_extract_dir_dict['clothing_dataset_grigorev']

       # clone repo and return to working dir 
        os.chdir('/')        
        os.system(f'git clone {source} {destination}')
        os.chdir(self.working_dir)

        return None

    def dataframes(self):
        """ Returns one dataframe conatining labels and image names """

        # go to root dir
        os.chdir('/')
        save_dir = self.local_zip_extract_dir_dict['clothing_dataset_grigorev']
        csv_file = os.path.join(save_dir, 'images.csv')

        # create dataframe & some cleanup
        df = pd.read_csv(csv_file)
        df = df.drop(columns=['sender_id', 'kids'])
        df = df[df['label']!= 'Not sure']

        # add image path info
        df['image_path'] = df['image'].apply(lambda x: os.path.join(save_dir, 'images', x))
        
        return df

