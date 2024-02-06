from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
from data.Mi2cai_Wrapper import TransWrapper, CocoMi2cai
from data.data_aug.data_aug import TRANSF_DICT, Sequence
import numpy as np
import pathlib, torch
from typing import Callable

############################## Collate funtion for dataloader

def _max_by_axis( the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def pad_and_create_pixel_mask(pixel_values_list, labels=None):
    """
    Pad images up to the largest image in a batch and create a corresponding :obj:`pixel_mask`.

    Args:
        pixel_values_list (:obj:`List[torch.Tensor]`):
            List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
        labels :
            The location of the bounday boxes must be normilized 

    Returns:
        - **pixel_values** -- Pixel values to be fed to a model.
        - **pixel_mask**   -- Pixel mask to be fed to a model.
        - **labels    **   -- adjusted labels
    """

    max_size = _max_by_axis([list(image.shape) for image in pixel_values_list])
    c, h, w = max_size
    padded_images = []
    pixel_mask = []
    for item, image in enumerate(pixel_values_list):
        # create padded image
        padded_image = np.zeros((c, h, w), dtype=np.float32)
        padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
        padded_images.append(padded_image)
        # create pixel mask
        mask = np.zeros((h, w), dtype=np.int64)
        mask[: image.shape[1], : image.shape[2]] = True
        pixel_mask.append(mask)
        # update labels. box [cx, cy, w, h ]
        if labels is not None:
            old_h, old_w = image.shape[1:]
            labels[item]["boxes"] = labels[item]["boxes"] * torch.tensor([[old_w/w, old_h/h, old_w/w, old_h/h]])
    padded_images = np.array(padded_images)
    pixel_mask = np.array(pixel_mask)
    if labels is not None:
        data = {"pixel_values": padded_images, "pixel_mask": pixel_mask, "labels":labels}
    else:  
        data = {"pixel_values": padded_images, "pixel_mask": pixel_mask}
    return data

def collate_fn(batch): # For handling Mi2cai data in data loader
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encoding = pad_and_create_pixel_mask(pixel_values, labels)
    batch = {}
    batch['pixel_values'] = torch.tensor(encoding['pixel_values'])
    batch['pixel_mask']   = torch.tensor(encoding['pixel_mask'])
    batch['labels']       = encoding['labels']
    return batch

def collate_R_MIS(batch): # For handling Robust MIS data in data loader 
    pixel_values = [item[0] for item in batch]
    labels = []
    for item in batch:
        item = torch.tensor(item[1])
        item = {"class_labels":item}
        labels.append(item)
    encoding = pad_and_create_pixel_mask(pixel_values)
    batch = {}
    batch['pixel_values'] = torch.tensor(encoding['pixel_values'])
    batch['pixel_mask']   = torch.tensor(encoding['pixel_mask'])
    batch['labels']       = labels
    return batch
    

#######################################################################

def trns_register(aug: dict) -> Callable:
    trfs = []
    for key in aug.keys():
        if (key != 'probability') and (aug[key] != None):
            if type(aug[key]) == tuple:
                trfs.append(TRANSF_DICT[key](*aug[key]))
            else:
                trfs.append(TRANSF_DICT[key](aug[key]))
    trfs = Sequence(trfs, aug['probability'])
    return TransWrapper(trfs)

def get_data(config, data_tag: str, shuffle=False):
    sets_path   = pathlib.Path(config.DATA.sets_path)
    transformation = None
    if data_tag == "train":
        root = config.DATA.train_imgs_path
        json_file = sets_path/config.DATA.train_set
        batch_size = config.TRAIN.train_batch
        if config.DATA.Do_augmentation:
            transformation = trns_register(config.DATA.AUGMENTATION)
            print("Transformations that will be applied on the training set:\n",transformation)
    #
    elif data_tag == "val":
        root = config.DATA.val_imgs_path
        json_file = sets_path/config.DATA.validation_set
        batch_size = config.TRAIN.val_batch
    elif data_tag == "test":
        root = config.DATA.test_imgs_path
        json_file = sets_path/config.DATA.test_set
        batch_size = config.TRAIN.test_batch
    else:
        raise ValueError ("Data type can be only one of the following {train,val,test}")
    
    feat_extractor = DetrImageProcessor(do_normalize=config.DATA.normalize,
        size=config.DATA.resize_shortedge, max_size=config.DATA.resize_longedge,
        image_mean=config.DATA.image_mean, image_std=config.DATA.image_std)
    
    dataset = CocoMi2cai(img_folder=root, annotation_file=json_file, remove_background=config.DATA.remove_background,
            feature_extractor=feat_extractor, transformations=transformation)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
    print(f"Number of samples for {data_tag} from {json_file}:", len(dataset))
    print(f"Batch size for {data_tag}: {batch_size}")

    return dataset,dataloader

######################################################
######           FREE FUNTIONS
######################################################
    
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b