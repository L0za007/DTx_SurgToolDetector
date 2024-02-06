##### Fix the warning for comming versions!!!
import warnings
warnings.filterwarnings("ignore")
##############################################
import pathlib
from config.config import get_cfg_defaults, combine_cfgs

from torch.utils.data import DataLoader
from data.Mi2cai_Wrapper import CocoMi2cai

from transformers import DetrImageProcessor
from tools import trns_register

feat_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
feat_extractor.size = {'shortest_edge': 550, 'longest_edge': 850}
    
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feat_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Test scrip for data augmentation in MICCAI-tool-2016")
    parser.add_argument("-c", "--configuration", type=str, default=None, help = "Path to experiment specifications (.YAML file)")
    args = parser.parse_args()
    if args.configuration:
        config = combine_cfgs(args.configuration)
    else:
        config = get_cfg_defaults()

    ######################################    Load dataset ############################################
    print("-----------------------------------\n",
    "#####\t Loading dataset",
    "\n-----------------------------------")
    root = config.DATA.train_imgs_path
    sets_path   = pathlib.Path(config.DATA.sets_path)
    train_jason = sets_path/config.DATA.train_set
    val_jason   = sets_path/config.DATA.validation_set

    if config.DATA.Do_augmentation:
        config.DATA.AUGMENTATION.probability = 1
        transformation = trns_register(config.DATA.AUGMENTATION)
        print("Transformations that will be applied on the training set:\n",transformation)
    else:
        transformation = None
    train_dataset = CocoMi2cai(img_folder=root, annotation_file=train_jason, feature_extractor=feat_extractor, transformations=transformation)
    print(f"Number of training samples from {train_jason}:", len(train_dataset))

    while True:
        print("Generating random transforms")
        for i in range(len(train_dataset)):
            _,_ = train_dataset[i]
