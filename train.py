
##### Fix the warning for comming versions!!!
import warnings
warnings.filterwarnings("ignore")
##############################################

import os,sys,pathlib
from datetime import datetime
from config.config import get_cfg_defaults, combine_cfgs

from tools import DETR_Wrapp, update_log_screen
from modeling.ModelTools import get_model

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

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

    if not (exp_path/"nets").exists():
        os.makedirs(exp_path/"nets")

    update_log_screen(config.OUTPUT_LOG, 'train_screen', 'w')
    print(f"Training of DETR model on {config.DATA.name}")
    print("Date: ", datetime.now())
    print(f"Used configuration: {config.name}")
    print(f"Folder results: {exp_path}, attempt: {config.OUTPUT_LOG.attept}")
    ######################################    Load dataset ############################################
    print("-----------------------------------\n",
    "#####\t Loading dataset",
    "\n-----------------------------------")
    if config.DATA.name == 'MICCAI16_tool_loc':
        from data.DataTools import get_data
        train_dataset, train_dataloader =  get_data(config, data_tag='train', shuffle=True)
        val_dataset  , val_dataloader   =  get_data(config, data_tag='val')
    elif config.DATA.name == 'SurgToolLoc_1126':
        from data.SurgToolLoc_wraper import get_data
        train_dataset, train_dataloader =  get_data(config, data_tag='train', shuffle=True)
        val_dataset  , val_dataloader   =  get_data(config, data_tag='val')


    update_log_screen(config.OUTPUT_LOG, 'train_screen')
    ######################################  Loss Criteria    ############################################
    print("-----------------------------------\n",
    "#####\t Loss criteria",
    "\n-----------------------------------")
    if config.TRAIN.loss_tags is None:
        print("Using default loss")
        criteria = None
    else:
        print("Using costum loss")
        from Solver.SolverTools import get_loss
        criteria = get_loss(config)
    ######################################   Model          ############################################
    print("-----------------------------------\n",
    "#####\t Model creation",
    "\n-----------------------------------")
    detector = get_model(config)
    model = DETR_Wrapp(detector, config, criteria)
    print("Model correctly initilized")

    update_log_screen(config.OUTPUT_LOG, 'train_screen')
    ######################################    Training       ############################################
    print("-----------------------------------\n",
    "#####\t Training",
    "\n-----------------------------------")
    stop_cfg = config.TRAIN.STOP_CRITERIA
    stop_criteria = EarlyStopping(monitor=stop_cfg.monitored_var, mode=stop_cfg.mode, 
                    patience=stop_cfg.patience, min_delta=stop_cfg.delta, verbose=True)

    trainer = Trainer(accelerator='gpu', devices=config.TRAIN.n_devices, 
                enable_progress_bar=True,
                max_epochs          = config.TRAIN.epochs, 
                gradient_clip_val   = 0.1,
                callbacks           = [stop_criteria], 
                check_val_every_n_epoch = config.TRAIN.check_val_every_n_epoch,
                default_root_dir    = exp_path)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path = config.TRAIN.checkpoint_path)
    ########################################################################################
    ######## Test for customized trainer
    ## trainer = Trainer(detector, config)
    ## print("SetUp for training completed")
    ## trainer.fit(tr_loader=train_dataloader, va_loader=val_dataloader, epochs=200)
    #########################################################################################
    ######################################    Saving results       ############################################
    print("-----------------------------------\n",
    "#####\t Saving final model",
    "\n-----------------------------------")

    if not (exp_path/"nets").exists():
        os.makedirs(exp_path/"nets")
    torch.save(detector.state_dict(), exp_path/"nets"/f"final{config.OUTPUT_LOG.attept}.pt")
