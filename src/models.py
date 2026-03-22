import torch
import torch.nn as nn
import geffnet # For EfficientNet
import timm    # For ViT
import os

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, load_pretrained_geffnet=False):
        super(enetv2, self).__init__()
        self.enet = geffnet.create_model(backbone.replace('-', '_'), pretrained=load_pretrained_geffnet)
        self.dropout = nn.Dropout(0.5) # Default, will be set from hparams
        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()
    
    def extract(self, x):
        return self.enet(x)
    
    def forward(self, x):
        x = self.extract(x)
        if x.dim() == 4: # EfficientNet specific GAP output handling
            x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.myfc(x)
        return x

class ViTFineTuner(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', out_dim=10, dropout_rate=0.5,
                 custom_pretrained_model_path=None, timm_imagenet_pretrained=False): # custom_pretrained_model_path is for backbone
        super().__init__()
        self.model_name = model_name
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate # Will be set from hparams

        # Determine if TIMM should load its own pretrained weights for the backbone
        use_timm_default_pretrained = timm_imagenet_pretrained
        if custom_pretrained_model_path and os.path.exists(custom_pretrained_model_path):
            print(f"  Custom pretrained ViT BACKBONE path ({custom_pretrained_model_path}) found. TIMM's default pretraining will be overridden.")
            use_timm_default_pretrained = False 

        print(f"  Creating TIMM model structure '{model_name}'. Initial TIMM pretraining for backbone: {use_timm_default_pretrained}")
        self.vit_model = timm.create_model(model_name, pretrained=use_timm_default_pretrained)
        
        # Load custom backbone weights if path is provided
        if custom_pretrained_model_path and os.path.exists(custom_pretrained_model_path):
            print(f"  Loading custom pretrained ViT BACKBONE weights from: {custom_pretrained_model_path}")
            state_dict = torch.load(custom_pretrained_model_path, map_location='cpu') # Load to CPU first
            
            # Handle 'module.' prefix if model was saved with DataParallel or DDP
            if any(key.startswith('module.') for key in state_dict.keys()):
                print("    Removing 'module.' prefix from state_dict keys.")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load backbone weights, allowing for mismatches (e.g. if it's a full checkpoint)
            missing_keys, unexpected_keys = self.vit_model.load_state_dict(state_dict, strict=False)
            print(f"    Custom backbone weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            if unexpected_keys and not all('head' in k for k in unexpected_keys): # If unexpected keys are not just the head
                 print(f"    Potentially problematic unexpected keys: {[k for k in unexpected_keys if 'head' not in k][:5]}")
            if missing_keys:
                 print(f"    Missing keys (could be an issue if not just head): {missing_keys[:5]}")

        # Replace the head for the current task
        num_ftrs = None
        # Try to get in_features from the existing head, if it's a Linear layer
        if hasattr(self.vit_model, 'head') and isinstance(self.vit_model.head, nn.Linear):
            num_ftrs = self.vit_model.head.in_features
        elif hasattr(self.vit_model, 'fc') and isinstance(self.vit_model.fc, nn.Linear): # Some models use 'fc'
             num_ftrs = self.vit_model.fc.in_features
        # Fallback to num_features (common in timm) or embed_dim
        elif hasattr(self.vit_model, 'num_features') and self.vit_model.num_features > 0:
            num_ftrs = self.vit_model.num_features
        elif hasattr(self.vit_model, 'embed_dim'):
            num_ftrs = self.vit_model.embed_dim
        else:
            raise AttributeError(f"Cannot determine input features for head of ViT model {model_name}")
        
        self.vit_model.head = nn.Linear(num_ftrs, self.out_dim) # Replace/set the head
        print(f"  Replaced/Set ViT head: In={num_ftrs}, Out={self.out_dim}")
        
        self.dropout = nn.Dropout(self.dropout_rate) # Dropout will be set by hparams

    def forward(self, x):
        # ViT typically uses the [CLS] token's features for classification
        features = self.vit_model.forward_features(x) 
        # For many timm ViTs, forward_features returns the sequence of patch embeddings + CLS token
        # The CLS token is usually the first one.
        cls_token_feature = features[:, 0] 
        
        cls_token_feature_dropped = self.dropout(cls_token_feature)
        logits = self.vit_model.head(cls_token_feature_dropped)
        return logits

if __name__ == '__main__':
    print("This module provides model definitions: enetv2 and ViTFineTuner.")
    # Example instantiation (won't run without data/hparams)
    # test_enet = enetv2("tf_efficientnet_b0", 10, load_pretrained_geffnet=True)
    # print("Enet instantiated.")
    # test_vit = ViTFineTuner("vit_tiny_patch16_224", 10, timm_imagenet_pretrained=True)
    # print("ViT instantiated.")