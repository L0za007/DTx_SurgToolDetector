import torch
from tools import time_it
import numpy as np

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)#

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

class FeatureLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @time_it
    def forward(self, feat, contrast, labels):
        ## Taking the only feature map that is used in the transformer
        f_1,m_1 = feat[-1]
        f_2,m_2 = contrast[-1]
        print(f_1.shape)
        print(f_2.shape)
        #print(labels)
        #Step 1 detect the features that will be compared
        height = f_1.shape[2] # 15
        width  = f_1.shape[3] # 26
        i_index = np.repeat(np.arange(height), width) ## Y
        j_index =   np.tile(np.arange(width), height) ## X
        #masks = []
        feat_loc = []
        selected_feat = []
        for i,item in enumerate(labels): ## Go through each element in the batch
            ## Rescale all the boxes of a single sample
            boxes = rescale_bboxes(item['boxes'], (width,height)).numpy() # each box [x,y,x2,y2]
            ## Generate mask that tell us which features to compare
            mask  = [(np.trunc(boxes[i,1])<=i_index)&(i_index<=boxes[i,3])&
                     (np.trunc(boxes[i,0])<=j_index)&(j_index<=boxes[i,2]) 
                     for i in np.arange(boxes.shape[0])]
            mask = np.any(mask, axis=0) if len(mask)>1 else mask[0]
            #masks.append(mask)
            feat_loc.append((i_index[mask],j_index[mask]))
            selected_feat.append((f_1[i,:,i_index[mask], j_index[mask]], 
                                  f_2[i,:,i_index[mask], j_index[mask]]))

        #regresar un para binario con los puntos a detectar
        #Step 2 compare the features
        
        #### Code to visualize the selected features for comparison 
        #image_ids = [labels[i]['image_id']for i in range(len(labels))]
        #return image_ids, masks
        return feat_loc, selected_feat
    
