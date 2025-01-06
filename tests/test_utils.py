import torch
import yaml

from fairchem.core.models.base import HydraModel


def load_data_model(
    data_path: str = "tests/data/OC20_batch_3.pt",
    model_path: str = "tests/data/L2_H4_64_.yml",
):
    batch = torch.load(data_path)
    with open(model_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    model = HydraModel(**config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return batch.to(device), model.to(device)
