import time, pathlib, sys
import torch.cuda as Cuda

def time_it(function):
    """ Decorator to time the execution of a function"""
    def wrapper(*args, **kwargs):
        t1 = time.time()
        results = function(*args, **kwargs)
        t2 = time.time()
        print(f"Finished {function.__name__!r} in {t2-t1:.4f} secs")
        return results
    return wrapper

def display_GPU_info(function):
    """ Decorator to display the GPU memory before and after the execution of a function"""
    def wrapper(*args, **kwargs):
        Cuda.empty_cache()
        print(f"Cuda device {Cuda.current_device()}")
        print("BEFORE FUNCTION EXCECUTION")
        print(f"Memory allocated {Cuda.memory_allocated()/1024**2 :5.3f}")
        print(f"Memory reserved  {Cuda.memory_reserved()/1024**2 :5.3f}")
        output = function(*args, **kwargs)
        print("AFTER FUNCTION EXCECUTION")
        print(f"Memory allocated {Cuda.memory_allocated()/1024**2 :5.3f}")
        print(f"Memory reserved  {Cuda.memory_reserved()/1024**2 :5.3f}")
        print(f"=========================================================")
        #time.sleep(5)
        return output
    return wrapper

def update_log_screen(config_log, file_name = 'log_screen', mode = 'a'):
    """ Update the log screen to save the output of the model"""
    if config_log.save_log:
        exp_path = pathlib.Path.cwd() / config_log.path / config_log.exp_tag
        log_file = f"{file_name}{config_log.attept}.txt"
        screen_path = exp_path/log_file
        if mode == 'a':
            sys.stdout.close()  
        f = open(screen_path,mode)
        sys.stderr = f
        sys.stdout = f

def process_output(outputs, threshold=0.5):
    """ Process the output of the model to extract the confidence 
    score, the labels and the bounding boxes
    Args:
        outputs (dict): Output of the model with the logits, predicted
            bounding boxes, last hidden state and encoder last hidden state
        threshold (float): Threshold to filter the predictions
    Returns:
        output (dict): Dictionary with the confidence score, 
            the labels and the bounding boxes
    """
    confidence = outputs.logits.softmax(-1)[:, :, :-1]
    boxes = outputs.pred_boxes
    max, indices = confidence.max(-1)
    keep = max > threshold
    output = []
    for i in range(confidence.shape[0]):
        prediction = {"p_score": max[i,keep[i,:]], 
                       "p_class": indices[i,keep[i,:]], 
                       "p_boxes":  boxes[i,keep[i,:]]
                       }
        ## Add the loss values if they are present. It is a batch loss 
        if "loss_dict" in outputs:
            prediction.update({"loss": outputs.loss})
            for k,v in outputs.loss_dict.items():
                prediction.update({k:v})
        output.append(prediction)
    return output
    

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as torch_TF

class DETR_Wrapp(pl.LightningModule):
    """ Wrapper for the DETR model to use it with pytorch lightning"""
    def __init__(self, model, config, criteria = None):
        """ Initialize the wrapper
        Args:
            model (DETR): DETR-based model
            config (dict): Configuration dictionary
            """
        super().__init__()
        self.model         = model
        self.weight_decay  = config.TRAIN.weight_decay
        self.l_rate        = config.TRAIN.learning_rate
        self.l_scheduler   = config.TRAIN.LEARNING_SCHEDULER
        self.results_log   = config.OUTPUT_LOG
        self.epoch_count   = config.TRAIN.last_epoch
        self.checkpoints   = config.TRAIN.last_manual_checkpoint
        self.criteria      = criteria
        self.loss_weight   = {k:v for k,v in zip(config.TRAIN.loss_components, config.TRAIN.loss_weights)}

        self.losses = {'Train':{'nb':1}, 'Val':{'nb':1}}
        self.val_flag = False

    def forward(self, pixel_values, pixel_mask, labels=None):
        """ Forward pass of the model
        Args:
            pixel_values (tensor): Tensor with the pixel values
            pixel_mask (tensor): Tensor with the pixel mask
            labels (list): List with the labels
        Returns:
            outputs (dict): Dictionary with the output of the model
        """
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs
    
    #@time_it
    def common_step(self, batch, batch_idx):
        """ Common step for the training and validation steps
        Args:
            batch (dict): Dictionary
            batch_idx (int): Index of the batch
        Returns:
            loss (tensor): Average loss value
            loss_dict (dict): Dictionary with the individual loss values
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
        # if no costum loss is defined, the model returns cross_entropy loss, boxes_loss {loss_giou + L1_norm_loss}, cardinality_loss
        if self.criteria is None:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss_dict = outputs.loss_dict
            loss = outputs.loss
        else: 
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            loss_dict = self.criteria(outputs, labels)
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight.keys())
            
        
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)   
        self._update_loss(loss, loss_dict, 'Train')

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx) 
        self._update_loss(loss, loss_dict, 'Val')

        self.val_flag = True
        return loss

    def configure_optimizers(self):
        """ Configure the optimizer and the learning rate scheduler
        Returns:
            optimizer (torch.optim): Optimizer
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        """
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],"lr": 1e-5,},]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.l_rate, weight_decay=self.weight_decay)
        
        if self.l_scheduler.fix_step:
            print(f"Used learning scheduler: STEP_LEARNING_RATE with factor {self.l_scheduler.factor}")
            scheduler_Step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.l_scheduler.step_size, gamma=self.l_scheduler.factor)
            return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler_Step}}
        else:
            print(f"Used learning scheduler: REDUCE_LR_ON_PLATEAU with factor {self.l_scheduler.factor}")
            scheduler_OP = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    factor=self.l_scheduler.factor , patience=self.l_scheduler.patience,
                    threshold=self.l_scheduler.delta, cooldown=self.l_scheduler.cooldown,
                    min_lr=self.l_scheduler.min_lr, verbose=True)
            return {"optimizer":optimizer, "lr_scheduler":{
            "scheduler": scheduler_OP,"monitor": self.l_scheduler.monitored_var}}

    def training_epoch_end(self, outputs):
        """ Called at the end of the training epoch. 
        We added this function just to print the loss values
        """
        self.epoch_count +=1
        print(f"Epoch {self.epoch_count:3.0f} completed.")
        print(self._get_loss_str('Train'), '| ', 
              self._get_loss_str('Val') if (self.val_flag and (self.epoch_count>2)) else "")
        
        self.losses['Train'] = {'nb':1}
        self.losses['Val'] = {'nb':1}
        self.val_flag = False
        update_log_screen(self.results_log, 'train_screen')

    def _get_loss_str(self, l_type):
        """ Get the string with the loss values from the log dictionaries"""
        loss_str = f"{l_type}: "
        nb = self.losses[l_type]['nb']
        for k,v in self.losses[l_type].items():
            if k != 'nb':
                loss_str += f"{k} {v/nb:3.3f} |"
        return loss_str

    def _update_loss(self, loss, loss_dict, l_type=None):
        """ Update the loss values in the log dictionaries and in the 
        tensorboard log. The tensorboard log is used to in the scheduler 
        and early stopping. The log dictionaries are used to print the 
        loss values on the screen."""
        if l_type == None:
            raise ValueError("Type of loss not specified")
        ## Log for tensorboard
        self.log(l_type+"_loss", loss.item())
        for k,v in loss_dict.items():
          self.log(l_type+"_"+k, v.item())

        ## Log to print on the screen log
        if 'loss' in self.losses[l_type].keys():
            self.losses[l_type]['loss'] +=loss.item()
        else:
            self.losses[l_type]['loss'] = loss.item()

        for k,v in loss_dict.items():
            if k in self.losses[l_type].keys():
                self.losses[l_type][k] += v.item()
            else:
                self.losses[l_type][k] = v.item()
        
        ## Update the number of batches
        self.losses[l_type]['nb'] +=1
