import torch
import timm
from huggingface_hub import login

# Define a function to load the model
def load_UNI2h(device: str = 'cuda'):
    """Load the UNI2 model from the huggingface hub
    
    Args:
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cuda'.
    Returns:
        VisionTransformer: The UNI2-h model.
    """
    # Read the huggingface token from a file
    with open("huggingface_token.key", "r") as f:
        token = f.read()

    # Login to huggingface hub with the token
    login(token)

    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
                'model_name': 'hf-hub:MahmoodLab/UNI2-h',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    
    # Create the model
    model = timm.create_model(pretrained=True, **timm_kwargs)

    # Put the model on the device
    model = model.to(device)
    
    return model