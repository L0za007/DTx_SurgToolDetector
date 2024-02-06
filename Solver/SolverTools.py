from Solver.base import DetrLoss
from transformers.models.detr.modeling_detr import DetrHungarianMatcher

def get_loss(config):
    print("Selected loss: ", config.TRAIN.loss_tags)
    n_classes = config.DATA.num_classes
    loss_tags = config.TRAIN.loss_tags
    h_matcher = DetrHungarianMatcher(class_cost=1, bbox_cost=5, giou_cost=2)
    criteria = DetrLoss(matcher=h_matcher, num_classes=n_classes, eos_coef=0.1, losses=loss_tags)
    w_per_component = {k:v for k,v in zip(config.TRAIN.loss_components, config.TRAIN.loss_weights)}
    print("Weight per component", w_per_component)
    return criteria
