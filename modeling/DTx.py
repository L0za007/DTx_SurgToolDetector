import torch
from transformers.models.detr.modeling_detr import DetrModelOutput
import math

class SinePositionEmbedding(torch.nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = int(embedding_dim//2)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2
        self.scale = scale * math.pi
        #print(f"Position embedding scale: {self.scale:3.3f}")

    def forward(self, pixel_values, pixel_mask):
        assert pixel_mask is not None, "No pixel mask provided"
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            #print("Normalizing position embedding.")
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embedding_dim)
        #print(dim_t.shape, dim_t)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class dense_input_proj(torch.nn.Module):
    def __init__(self, feat_extractor, hidden_dim, selected_maps, activation=False) -> None:
        super().__init__()
        self.selected_maps = selected_maps
        feat_maps_size = self.get_feat_maps_size(feat_extractor)
        #print(feat_maps_size)
        self.ConvProjections = torch.nn.ModuleList([
            torch.nn.Conv2d(feat_maps_size[map][0],hidden_dim, kernel_size=1) 
            for map in selected_maps])
        if activation:
            print("Using Sigmoid activation in projection layer.")
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = None
        
    def forward(self, feat_maps):
        projected_maps = []
        for pro_idx, map_idx in enumerate(self.selected_maps):
            map, mask = feat_maps[map_idx]
            projection = self.ConvProjections[pro_idx](map)
            if self.activation is not None:
                projection = self.activation(projection)
            projected_maps.append((projection, mask))
            #print(map.shape)
        
        return projected_maps

    def get_feat_maps_size(self, feat_extractor):
        layer_shape= []
        layer_shape.append(feat_extractor.layer1[-1].conv3.weight.shape)
        layer_shape.append(feat_extractor.layer2[-1].conv3.weight.shape)
        layer_shape.append(feat_extractor.layer3[-1].conv3.weight.shape)
        layer_shape.append(feat_extractor.layer4[-1].conv3.weight.shape)
        return layer_shape


class DTX(torch.nn.Module):
    def __init__(self, model, exp_config) -> None:
        super().__init__()
        self.config = model.config
        # Backbone + positional encoding
        normalize_position = exp_config.MODEL.normalize_position
        position_scale = exp_config.MODEL.position_scale
        backbone  = model.backbone
        backbone.position_embedding = SinePositionEmbedding(normalize=normalize_position, scale=position_scale)
        self.backbone =  backbone

        # Projection layer
        # input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)
        self.feature_selection = exp_config.MODEL.feature_selection
        self.input_projection = dense_input_proj(backbone.conv_encoder.model, 
            model.config.d_model, self.feature_selection, activation=exp_config.MODEL.add_pos_activation)

        # nn.Embedding(config.num_queries, config.d_model)
        # The number of query is modified in the get_model funtion from ModelTools.py no need to change it here
        self.query_position_embeddings = model.query_position_embeddings

        # Transformer
        # The configuration of the transformer is modified in the get_model funtion from ModelTools.py no need to change it here
        self.encoder = model.encoder
        self.decoder = model.decoder
        
        # Original implementation apply a costum initialization of weights
        
    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default) of each feature map
        projected_feature_maps = self.input_projection(features)

        # Third, flatten the feature maps and position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # Also flatten the mask to NxHW
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = []
        flattened_masks = []
        for i in range(len(projected_feature_maps)):
            flattened_features.append(projected_feature_maps[i][0].flatten(2).permute(0, 2, 1))
            flattened_masks.append(projected_feature_maps[i][1].flatten(1))

        position_embeddings = []
        for i in self.feature_selection:
            position_embeddings.append(position_embeddings_list[i].flatten(2).permute(0, 2, 1))

        # Fourth, concatenate the flattened feature maps and position embeddings along the sequence length dimension
        # Also concatenate the masks along the sequence length dimension
        flattened_features = torch.cat(flattened_features, dim=1)
        flattened_masks = torch.cat(flattened_masks, dim=1)
        position_embeddings = torch.cat(position_embeddings, dim=1)
        
        encoder_outputs = self.encoder(
            inputs_embeds=flattened_features,
            attention_mask=flattened_masks,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )

