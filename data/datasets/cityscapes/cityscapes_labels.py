from collections import namedtuple
import numpy as np
import torch
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label,
    'wilddashId',   # Wilddash eval class id
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color           wilddash class id
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),     25 ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),     0 ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),     25 ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),     25 ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0),     25 ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0),     25 ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81),     25 ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128),     1 ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232),     2 ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160),     25 ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140),     25 ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70),     3 ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156),     4 ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153),     5 ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180),     6 ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100),     25 ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90),     25 ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153),     7 ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153),     25 ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30),     8 ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0),     9 ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35),     10 ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152),     11 ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180),     12 ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60),     13 ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0),     14 ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142),     15 ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70),     16 ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100),     17 ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90),     25 ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110),     25 ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100),     25 ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230),     18 ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32),     19 ),
    Label(  'license plate'        , 34 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142),     25 ),
]

def create_id_to_name():
    id_to_name = {}
    for lbl in _labels:
        if lbl[2] != 255:
            id_to_name[lbl[2]] = lbl[0]
    del id_to_name[-1]
    id_to_name[19] = 'ignore'
    return id_to_name

def create_name_to_id():
    name_to_id = {}
    for lbl in _labels:
        if lbl[2] != 255:
            name_to_id[lbl[0]] = lbl[2]
    del name_to_id['license plate']
    name_to_id['ignore'] = 19
    return name_to_id

def create_id_to_train_id_mapper():
    lookup_table = torch.ones(len(_labels)-1).long() * 19
    for label in _labels[:-1]:
        lookup_table[label[1]] = label[2]
    lookup_table[lookup_table == 255] = 19
    return lookup_table

def create_osr_id_to_train_id_mapper():
    lookup_table = torch.ones(len(_labels)).long() * 20
    for label in _labels[:-1]:
        lookup_table[label[1]] = label[2]
    lookup_table[lookup_table == 255] = 20
    lookup_table[-1] = 19
    return lookup_table

class ColorizeLabels:
    def __init__(self):
        color_info = [label.color for label in _labels if label.ignoreInEval is False]
        color_info += [[0, 0, 0]]
        self.color_info = np.array(color_info)

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0]
            G[mask] = self.color_info[l][1]
            B[mask] = self.color_info[l][2]
        return torch.LongTensor(np.stack((R, G, B), axis=-1).astype(np.uint8)).squeeze().permute(2, 0, 1).float() / 255.

    def __call__(self, example):
        return self._trans(example)

colorize_labels = ColorizeLabels()

mean, std = torch.tensor([0.2869, 0.3251, 0.2839]).view(3, 1, 1), torch.tensor([0.1761, 0.1810, 0.1777]).view(3, 1, 1)
def denormalize_city_image(img):
    img = img * std.to(img)
    return img + mean.to(img)


class ColorizeOSRLabels:
    def __init__(self):
        color_info = [label.color for label in _labels if label.ignoreInEval is False]
        color_info += [[255, 255, 255]]
        color_info += [[0, 0, 0]]
        self.color_info = np.array(color_info)

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0]
            G[mask] = self.color_info[l][1]
            B[mask] = self.color_info[l][2]
        return torch.LongTensor(np.stack((R, G, B), axis=-1).astype(np.uint8)).squeeze().permute(2, 0, 1).float() / 255.

    def __call__(self, example):
        return self._trans(example)

colorize_osr_labels = ColorizeOSRLabels()

if __name__ == '__main__':
    # print(len(create_id_to_name()))
    print(create_id_to_train_id_mapper())




