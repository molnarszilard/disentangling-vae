import json
import os
import re

import torch

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from utils.datasets import get_img_size


MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"  # CHANGE TO METADATA.json


def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to the directory where to save the data.

    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    path_to_metadata = os.path.join(directory, META_FILENAME)
    path_to_model = os.path.join(directory, filename)

    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device

    if metadata is not None:
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)


def load_metadata(directory):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    """
    path_to_metadata = os.path.join(directory, META_FILENAME)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_model = os.path.join(directory, MODEL_FILENAME)
    metadata = load_metadata(directory)

    dataset = metadata["dataset"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]
    img_size = get_img_size(dataset)

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model


def load_checkpoints(directory, is_gpu=True):
    """Load all chechpointed models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints


def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Get model """
    encoder = get_Encoder(model_type)
    decoder = get_Decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    return model