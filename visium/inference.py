#!/opt/conda/bin/python

import numpy as np
import pandas as pd
import torch
import openslide
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import Visium_HES_Dataset
from model import load_UNI2h

# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the UNI2 model
model = load_UNI2h(device)

# Create the transform function from the model configuration
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# Determine the slides id
path = "data_hdd/visium/results"
slides = os.listdir(path)

for slide_id in slides:
    # Check if the embedding features already exist
    if not os.path.exists(f"{path}/{slide_id}/hes_features.pkl"):
        print(f"Processing {slide_id}")

        # Load the slide with openslide
        slide = openslide.OpenSlide(f"{path}/{slide_id}/images/annotation_images/visium_region.ome.tiff")

        # Load the pixels of the MALDI MSI image
        coordinates = pd.read_csv(f"{path}/{slide_id}/coordinates_warped.csv")

        # Create the dataset
        dataset = Visium_HES_Dataset(slide=slide,
                                     coordinates=coordinates,
                                     transform=transform)

        # Create the dataloader
        dataloader = DataLoader(dataset,
                                batch_size=256,
                                shuffle=False)

        # Create the embedding
        feature_emb = []

        # Iterate over the dataloader with tqdm
        for batch in tqdm(dataloader, desc="Processing batches"):
            
            # Get the embedding from the model
            with torch.inference_mode():
                output = model(batch.to(device))
            
            # Append the embedding to the list
            feature_emb.append(output.cpu().numpy())

        # Concatenate the embeddings
        feature_emb = np.concatenate(feature_emb)

        # Clean up the GPU memory
        torch.cuda.empty_cache()

        # Transform the embedding features into a dataframe
        feature_emb_df = pd.DataFrame(feature_emb, index=coordinates.index)

        # Save the embedding features to a pickle file
        feature_emb_df.to_pickle(f"{path}/{slide_id}/hes_features.pkl")