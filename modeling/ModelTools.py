from collections import OrderedDict
from modeling.Res2Net import Res2Net, Bottle2neck
from modeling.Res2Net import res2net50_config

def change_backbone(model, config):
    if config.add_multi_scale_backbone:
        print("Adding Res2Net multiscale blocks")
        if config.MS_config == "custom":
            print("Custom Rest2Net50 [3,4,6,3], baseWidth=16,scale=4. This model doesn't have pre-trained weights")
            encoder = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 16, scale = 4)
        else:
            print(f"Using Res2Net {config.MS_config}")
            encoder = res2net50_config[config.MS_config](pretrained=config.pretrained, BottleNeck=Bottle2neck)
        
    model.conv1  = encoder.conv1
    model.bn1    = encoder.bn1
    model.layer1 = encoder.layer1
    model.layer2 = encoder.layer2
    model.layer3 = encoder.layer3
    model.layer4 = encoder.layer4
    

def load_wieghts_coder(new_coder, trained_coder, num_layers=6):
    """Function to load weights from a trained encoder or decoder to a new 
    transformer starting from the final layer and going backwards"""
    for i in range (num_layers):  ## I substract 1 because the layers are indexed from 0
        coder_params = OrderedDict([(name,params) for name, params
            in trained_coder.layers[6-1-i].state_dict().items()])
        mk,uk = new_coder.layers[num_layers-1-i].load_state_dict(coder_params, strict=False)
    return new_coder

def update_transformer(detector, TX_config):
    detr_cfg = DetrConfig(decoder_layers =TX_config.decoder_layers, 
                          encoder_layers =TX_config.encoder_layers,
                                 d_model =TX_config.hidden_dim)
    modified_model =DetrModel(detr_cfg)
    # Changing the number of layers in the transformer
    print("Changing decoder layers to ", TX_config.decoder_layers,
          " and encoder layers to ", TX_config.encoder_layers)
    detector.config.decoder_layers = TX_config.decoder_layers
    detector.config.encoder_layers = TX_config.encoder_layers

    # Loading weights from the trained model
    if TX_config.load_weights:
        print("Loading weights from trained transformer")
        detector.model.encoder = load_wieghts_coder(
            modified_model.encoder, detector.model.encoder, TX_config.encoder_layers)
        detector.model.decoder = load_wieghts_coder(
            modified_model.decoder, detector.model.decoder, TX_config.decoder_layers)
    


from transformers import DetrForObjectDetection, DetrConfig, DetrModel
from modeling.DTx import DTX

def get_model(config):
    detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
            num_labels=config.DATA.num_classes, ignore_mismatched_sizes=True)
    
    if not config.MODEL.pretrained:
        weights_cfg = DetrConfig()
        print("Reinitializing all weights...")
        detector.model = DetrModel(weights_cfg)
    
    for _, param in detector.named_parameters():
        param.requires_grad = True
    
    if config.MODEL.add_multi_scale_backbone:
        print("Changing backbone config.")
        change_backbone(detector.model.backbone.conv_encoder.model, config.MODEL)
    
    if config.MODEL.TX.custom:
        print("Changing transformer config.")
        update_transformer(detector, config.MODEL.TX)

    if config.MODEL.dense_inference:
        print("Adding connections for dense inference.")
        print(f"Used feature maps are {config.MODEL.feature_selection}")
        dense_model = DTX(detector.model, config)
        detector.model = dense_model

    return detector

