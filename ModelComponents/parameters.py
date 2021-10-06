# imports
import tensorflow as tf
import os
import sys

"""
This contains some model parameters to ensure consistency between model components
and the model implementation.
"""


class Filepaths:
    def __init__(self, model_name='custom_DETR', dataset_name='Fashionpedia', drive_type='google_drive'):

        self._dataset_name = dataset_name
        self._model_name = model_name
        
        # filepaths
        if drive_type == 'google_drive':
            archive_base_dir = '/content/drive/MyDrive/datasets/'
            local_base_dir = '/content/datasets/'
            checkpoint_dir = os.path.join(archive_base_dir, self._dataset_name, 'ModelCheckpoints/')
            model_files_dir = '/content/drive/MyDrive/GitHub/ShuffleExchange/Shuffle_AI/ModelComponents/'
            tfrec_files_dir = os.path.join('gs://shuffle-datasets/', self._dataset_name, 'tfrec/')

        elif drive_type == 'google_cloud_storage':
            archive_base_dir = 'gs://shuffle-datasets/'       
            local_base_dir = 'gs://shuffle-datasets/'
            checkpoint_dir = 'gs://shuffle-datasets/ModelCheckpoints/'
            model_files_dir = 'gs://shuffle-datasets/ModelComponents/'
            tfrec_files_dir = os.path.join('gs://shuffle-datasets/', self._dataset_name, 'tfrec/')
            
        else:
            archive_base_dir = input('Enter data archive path: ')
            local_base_dir = input('Enter data extraction path: ')
            checkpoint_dir = input('Enter checkpoint_dirdirectory: ')
            model_files_dir = input('Enter model files directory: ')
            tfrec_files_dir = input('Enter tfrec files directory: ')

        self._archive_base_dir = archive_base_dir
        self._local_base_dir = local_base_dir
        self._checkpoint_dir = checkpoint_dir
        self._model_files_dir = model_files_dir
        self._tfrec_files_dir = tfrec_files_dir

        # access attributes
    def default_params(self, value=None):
        parameters = {'dataset_name': self._dataset_name,
                      'model_name':self._model_name,
                      'archive_base_dir': self._archive_base_dir,
                      'local_base_dir': self._local_base_dir,
                      'checkpoint_dir': self._checkpoint_dir,
                      'model_files_dir': self._model_files_dir,
                      'tfrec_files_dir': self._tfrec_files_dir}
        
        if value is not None:
            return parameters[value]
        else:
            return parameters

#class MixedPrecision:
class StrategyOptions:    
    def __init__(self, mixed_precision=True):

        # mixed precision
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() 
            STRATEGY = tf.distribute.TPUStrategy(tpu)
            MIXED_PRECISION_TYPE = 'mixed_bfloat16' 
            os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"  # for TF Hub models on TPU

            print('Loaded TPU')

        except ValueError:           
            if tf.config.list_physical_devices('GPU'):  # check for GPU
                MIXED_PRECISION_TYPE = 'mixed_float16'
                STRATEGY = tf.distribute.MirroredStrategy() # single-GPU or multi-GPU
                
            else:  # CPU
                MIXED_PRECISION_TYPE = None
                STRATEGY = tf.distribute.get_strategy()
        
        if not mixed_precision:
            MIXED_PRECISION_TYPE = None

        tf.keras.mixed_precision.set_global_policy(MIXED_PRECISION_TYPE)
        self._mixed_precision = MIXED_PRECISION_TYPE
        self._strategy = STRATEGY
    
    # access attributes
    def strategy(self):
        return self._strategy
    def precision(self):
        return self._mixed_precision


# Set Model Parameters
class ModelParameters:
    def __init__(self, dataset_name='Fashionpedia'):

        # constants
        self._num_object_preds = 96
        self._image_size = (560, 560)  # shape to resize images in data pipeline. 

        # required constants.  Do not change these!
        self._pad = '<PAD>'  # pad token. Do not change!
        self._oov = '<OOV>'  # out of vocabulary token. Do not change!

        # default dataset
        self._dataset_name = dataset_name

    # functions to access params
    def dataset_name(self):
        return self._dataset_name    
        
    def vocab_dict(self):
        Fashionpedia_vocab_dict = {
            'attribute': ['sweatpants', 'dolman (sleeve), batwing (sleeve)', 'ringer (t-shirt)', 'high low', 'fur', 'single breasted', 'trucker (jacket)', 'skater (dress)', 'hip-huggers (pants)', 'flare', 'wrap (skirt)', 'chevron', 'giraffe', 'tulip (skirt)', 'v-neck', 'double breasted', 'gathering', 'pleat', 'flap (pocket)',
                        'puffer (jacket)', 'zebra', 'toile de jouy', 'metal', 'anorak', 'micro (length)', 'accordion (skirt)', 'puff (sleeve)', 'sheath (skirt)', 'bell (sleeve)', 'duffle (coat)', 'nehru (jacket)', 'cheetah', 'three quarter (length)', 'peacock', 'peasant (top)', 'no waistline', 'jodhpur', 'round (neck)', 'surplice (neck)', 'curved (fit)',
                        'bead(a)', 'stand-away (collar)', 'cargo (skirt)', 'asymmetrical', 'patch (pocket)', 'bermuda (shorts)', 'kaftan', 'short (shorts)', 'chemise (dress)', 'sweetheart (neckline)', 'smock (top)', 'stripe', 'chained (opening)', 'snakeskin (pattern)', 'normal waist', 'gown', 'jeans', 'maxi (length)', 'peak (lapel)', 'jabot (collar)', 'slit',
                        'turtle (neck)', 'dirndl (skirt)', 'snakeskin', 'peg', 'teddy bear (coat)', 'sarong (skirt)', 'military (coat)', 'elbow-length', 'floor (length)', 'shirt (dress)', 'paisley', 'leg of mutton (sleeve)', 'cheongsams', 'embossed', 'track (pants)', 'lace up', 'tutu (skirt)', 'hobble (skirt)', 'feather', 'booty (shorts)', 'wood',
                        'sailor (collar)', 'trunks', 'knee (length)', 'cap (sleeve)', 'sailor (pants)', 'crossover (neck)', 'sailor (shirt)', 'robe', 'leopard', 'puffer (coat)', 'letters, numbers', 'norfolk (jacket)', 'sundress', 'empire waistline', 'oversized', 'wrapping', 'nightgown', 'hoodie', 'kimono', 'roll-up (shorts)', 'high waist', 'bootcut', 'toggled (opening)',
                        'ruched', 'wrist-length', 'mao (jacket)', 'tulip (sleeve)', 'blouson (dress)', 'tuxedo (jacket)', 'halter (dress)', 'notched (lapel)', 'square (neckline)', 'sweater (dress)','banded (collar)', 'cowl (neck)', 'dropped waistline', 'prairie (skirt)', 'buckled (opening)', 'dropped-shoulder sleeve', 'check', 'symmetrical', 'crop (jacket)', 'below the knee (length)',
                        'duster (coat)', 'leggings', 'suede', 'fit and flare', 'polo (shirt)', 'henley (shirt)', 'halter (neck)', 'plain (pattern)', 'cargo (pocket)', 'no opening', 'capri (pants)', 'floral', 'lounge (shorts)', 'smocking', 'blanket (coat)', 'baggy', 'safari (jacket)', 'poet (sleeve)', 'basque (wasitline)','perforated', 'no non-textile material', 'peg (pants)', 'chelsea (collar)',
                        'asymmetric (neckline)', 'bloomers', 'short (length)', 'collarless', 'bodycon (dress)', 'blazer', 'wrap (dress)', 'mini (length)', 'kimono (sleeve)', 'fly (opening)','plant', 'oversized (lapel)', 'shift (dress)', 'sleeveless', 'shawl (lapel)', 'tunic (dress)', 'curved (pocket)', 'halter (top)', 'houndstooth (pattern)', 'crop (pants)', 'high (neck)', 'balloon', 'seam (pocket)',
                        'culottes', 'straight across (neck)', 'geometric', 'set-in sleeve', 'fair isle', 'tie-up (shorts)', 'swing (coat)', 'pea (jacket)', 'harem (pants)', 'culotte (shorts)', 'camo (pants)', 'wrap (coat)', 'loose (fit)', 'slip (dress)', 'tea (dress)', 'camouflage', 'tank (top)', 'bell bottom', 'asymmetric (collar)', 'ivory', 'tight (fit)', 'circle', 'cargo (pants)', 'windbreaker',
                        'circular flounce (sleeve)', 'peter pan (collar)', 'kangaroo (pocket)', 'skater (skirt)', 'rubber', 'oversized (collar)', 'quilted', 'bow (collar)', 'godet (skirt)', 'regular (fit)', 'biker (jacket)', 'cargo (shorts)', 'gypsy (skirt)', 'shearling (coat)','crew (neck)', 'raglan (sleeve)','raincoat', 'oval (neck)', 'gem', 'bishop (sleeve)',
                        'argyle', 'flamenco (skirt)', 'polo (collar)', 'off-the-shoulder', 'no special manufacturing technique', 'varsity (jacket)', 'peplum', 'chanel (jacket)', 'trumpet', 'hip (length)', 'wide leg', 'washed', 'regular (collar)', 'bolero', 'zip-up', 'trench (coat)', 'slash (pocket)', 'kilt', 'crop (top)', 'scoop (neck)', 'illusion (neck)',
                        'herringbone (pattern)', 'above-the-hip (length)', 'rivet(a)', 'classic military (jacket)', 'printed', 'classic (t-shirt)', 'raglan (t-shirt)', 'dress (coat )', 'u-neck', 'keyhole (neck)', 'sequin(a)', 'burnout', 'napoleon (lapel)', 'crocodile', 'cartoon', 'pencil', 'bone', 'applique(a)', 'sheath (dress)', 'boardshorts', 'pea (coat)', 'mermaid',
                        'abstract', 'undershirt', 'shearling', 'midi', 'jumper (dress)', 'distressed', 'low waist', 'tube (top)', 'tiered', 'rugby (shirt)', 'welt (pocket)', 'rah-rah (skirt)', 'ball gown (skirt)', 'track (jacket)', 'bomber (jacket)', 'dot', 'straight', 'cutout', 'lining', 'boat (neck)', 'shirt (collar)', 'plunging (neckline)', 'above-the-knee (length)',
                        'frayed', 'tunic (top)', 'choker (neck)', 'tent', 'camisole', 'queen anne (neck)', 'one shoulder', 'bell', 'plastic', 'mandarin (collar)', 'a-line', 'parka', 'skort'],

            'category':['collar', 'skirt', 'bag, wallet', 'tie', 'buckle', 'bow', 'shoe', 'ruffle', 'headband, head covering, hair accessory', 'umbrella', 'zipper', 'vest', 'cardigan', 'shorts', 'bead', 'sock', 'jumpsuit', 'dress', 'cape', 'leg warmer', 'glasses', 'pocket', 'hood',
                        'scarf', 'shirt, blouse', 'rivet', 'glove', 'ribbon', 'sleeve', 'epaulette', 'tights, stockings', 'fringe', 'flower', 'tassel', 'neckline', 'top, t-shirt, sweatshirt', 'pants', 'sequin', 'sweater', 'coat', 'applique', 'belt', 'hat', 'lapel', 'jacket', 'watch']                     
        }

        _vocab_dict = {}
        _vocab_dict[self._dataset_name] = Fashionpedia_vocab_dict
        return _vocab_dict
        
    def default_vocab(self):
        vocab_dict = self.vocab_dict()
        dataset_name = self.dataset_name()
        return vocab_dict[dataset_name]
        
    def default_params(self, value=None):
        parameters = {'image_size': self._image_size,
                       'encoder_dim': 256,
                       'num_encoder_blocks': 4,
                       'num_encoder_heads': 8,
                       'num_decoder_blocks': 4,  # MUST be >= 1
                       'num_decoder_heads': 8,
                       'decoder_dim': 256,
                       'num_panoptic_heads': 1,
                       'panoptic_dim': 32,
                       'num_object_preds': self._num_object_preds,
                       'vocab_dict': self.default_vocab(),
                       'pad_value': self._pad,
                       'oov_value': self._oov,
        }
        if value is not None:
            return parameters[value]
        else:
            return parameters