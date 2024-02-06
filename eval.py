
##### Fix the warning for comming versions!!!
import warnings
warnings.filterwarnings("ignore")
##############################################

import os,sys,pathlib
from datetime import datetime
from config.config import get_cfg_defaults, combine_cfgs

from transformers import DetrImageProcessor
from tools import DETR_Wrapp, update_log_screen
from modeling.ModelTools import get_model
from data.DataTools import get_data

#from detr.datasets import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import torch


def evaluation(current_loader,model,device,feat_extractor):
    model.to(device)
    base_dataset = current_loader.dataset.coco
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_dataset, iou_types) # initialize evaluator with ground truths
    with torch.no_grad():
        for idx, batch in enumerate(current_loader):
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

            # forward pass
            outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = feat_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}
            coco_evaluator.update(res)
            del outputs

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    

def speed_test(config,model):
    model.to(device)
    config.DATA.remove_background = True
    config.DATA.resize_longedge = config.DATA.resize_shortedge
    print("input size: ",config.DATA.resize_shortedge,"X", config.DATA.resize_longedge)
    _, loader = get_data(config, data_tag='test')
    dataset = loader.dataset
    import time
    total_time = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            
            # forward pass
            t1 = time.time()
            outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            t2 = time.time()
            total_time += (t2-t1)
            del outputs
    
    print(f"Total time {total_time} for {len(dataset)} samples in {len(loader)} batches.\n",
          f"Time per image {total_time/len(dataset)}, time per batch {total_time/len(loader)}\n",
          f"Frames per second {len(dataset)/total_time}, batches per second {len(loader)/total_time}")
    total_time = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            pixel_values,_ = dataset[i]
            pixel_values =  pixel_values[None,:,:,:].to(device)
            # forward pass
            t1 = time.time()
            outputs = model.model(pixel_values=pixel_values)
            t2 = time.time()
            total_time += (t2-t1)
    print(f"Test on one by one image, total time {total_time} for {len(dataset)} images",
          f"\n FPS: {len(dataset)/total_time}\n seconds per image {total_time/len(dataset)}")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Training script for DETR on endoscopic images.")
    parser.add_argument("-c", "--configuration", type=str, default=None, help = "Path to experiment specifications (.YAML file)")
    args = parser.parse_args()
    if args.configuration:
        if pathlib.Path(args.configuration).exists():
            config = combine_cfgs(args.configuration)
        else:
            raise FileNotFoundError(args.configuration)
    else:
        config = get_cfg_defaults()

    exp_path = pathlib.Path.cwd() / config.OUTPUT_LOG.path / config.OUTPUT_LOG.exp_tag
    if not exp_path.exists():
        os.makedirs(exp_path)
    
    update_log_screen(config.OUTPUT_LOG, 'evaluation_screen', 'w')
        
    print(f"Evaluation of DETR model on {config.DATA.name}")
    print("Date: ", datetime.now())
    print(f"Used configuration: {config.name}")
    print(f"Folder results: {exp_path}")
    ######################################    Load dataset ############################################
    print("-----------------------------------\n",
    "#####\t Loading dataset",
    "\n-----------------------------------")
    if config.DATA.remove_background:
        config.DATA.remove_background = False
        print("Romove background Function is not available in eval mode")
    test_dataset, test_dataloader =  get_data(config, data_tag='test')
    val_dataset , val_dataloader  =  get_data(config, data_tag='val')

    feat_extractor = DetrImageProcessor(do_normalize=config.DATA.normalize,
        size=config.DATA.resize_shortedge, max_size=config.DATA.resize_longedge,
        image_mean=config.DATA.image_mean, image_std=config.DATA.image_std)

    update_log_screen(config.OUTPUT_LOG, 'evaluation_screen')
    ######################################   Model          ############################################
    print("-----------------------------------\n",
    "#####\t Model creation",
    "\n-----------------------------------")
    detector = get_model(config)
    model = DETR_Wrapp(detector, config)
    model_path = exp_path/"nets"/f'final{config.OUTPUT_LOG.attept}.pt'
    model.model.load_state_dict(torch.load(model_path))
    print(f"Model {model_path} was loaded corectly")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    update_log_screen(config.OUTPUT_LOG, 'evaluation_screen')
#########################################################################
###################   EVAL  #############################################
#########################################################################
    print("-----------------------------------\n",
    "#####\t Evaluation on Validationset",
    "\n-----------------------------------")
    evaluation(val_dataloader,model,device,feat_extractor)
    torch.cuda.empty_cache()

    update_log_screen(config.OUTPUT_LOG, 'evaluation_screen')
    print("-----------------------------------\n",
    "#####\t Evaluation on Testset",
    "\n-----------------------------------")
    evaluation(test_dataloader,model,device,feat_extractor)
    torch.cuda.empty_cache()

    update_log_screen(config.OUTPUT_LOG, 'evaluation_screen')
    print("-----------------------------------\n",
    "#####\t Speed test",
    "\n-----------------------------------")
    speed_test(config,model)
    