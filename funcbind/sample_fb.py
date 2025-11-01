from omegaconf import OmegaConf
from funcbind.train_fb import sample
from funcbind.utils.constants import is_antibody_dataset
from funcbind.utils.utils_nf import update_config_nf
import hydra
import torch
import os

from funcbind.utils.utils_fb import create_field_makers, load_funcbind_sampling
from funcbind.utils.utils_base import setup_fabric
from funcbind.utils.utils_dataset import create_field_loaders
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from funcbind.dataset.preprocess_single_pdb import preprocess_ligand_receptor

# Suppress INFO messages from botocore.httpchecksum
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class SingleItemDataset(Dataset):
    """Simple dataset wrapper for a single preprocessed ligand-pocket pair."""

    def __init__(self, ligands, pockets):
        self.ligands = ligands
        self.pockets = pockets

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "receptor": self.pockets,
            "ligand": self.ligands
        }


def single_item_collate_fn(batch):
    """Custom collate function that handles None ligand values."""
    if not batch:
        return None

    # Since SingleItemDataset always returns a single item, batch will have length 1
    item = batch[0]

    # Handle the case where ligand is None
    if item["ligand"] is None:
        # Create a batch with only receptor for default collate
        receptor_only_batch = [{"receptor": item["receptor"]}]
        result = default_collate(receptor_only_batch)
        result["ligand"] = None
        return result
    else:
        # Use default collate for normal case
        return default_collate(batch)


@hydra.main(config_path="configs", config_name="sample_fb_ab", version_base=None)
def main(config):
    # -----------------------------------------------------------
    # basic inits
    assert config["sampling"]["chain_init"] in ["denovo", "ligand"]
    assert torch.cuda.is_available(), "not a good idea to train on cpu..."

    fabric = setup_fabric(config)

    # load FuncBind model
    OmegaConf.resolve(config)
    model, enc, dec_module, config, config_nf, num_classes = load_funcbind_sampling(config, fabric)

    # loaders
    config["dset"]["batch_size"] = 1
    config_nf = update_config_nf(config_nf, config)

    # Check if external PDB file is specified (SDF is optional for antibody mode)
    if ("pdb_path" in config and config["pdb_path"] is not None):
        os.makedirs("dataset/data/test_raw_pdbs/", exist_ok=True)
        if is_antibody_dataset(config):
            modality = "ab"
            sdf_path = None
            fabric.print(f"Using external file in ANTIBODY mode: PDB={config['pdb_path']}")
        elif "mcpp" in config["dset"]["use_single_dataset"]:
            modality = "mcp"
            sdf_path = config["pdb_path"][:-len('-protein.pdb')] + '-CP.sdf' if config["pdb_path"] is not None else None
            if not os.path.exists(sdf_path):
                sdf_path = None
            fabric.print(f"Using external files in MCP mode: PDB={config['pdb_path']}, SDF={sdf_path}")
        else:
            modality = "sm"
            sdf_path = config["pdb_path"][:-len('_pocket10.pdb')] + '.sdf' if config["pdb_path"] is not None else None
            fabric.print(f"Using external files in SMALL MOLECULE mode: PDB={config['pdb_path']}, SDF={sdf_path}")

        # Use preprocess_ligand_pocket to process external files
        ligand, receptor = preprocess_ligand_receptor(
            target_pdb_file=config["pdb_path"],
            ligand_sdf_file=sdf_path,
            receptor_center=config.get("receptor_center", None),
            aug=False,
            out_dir="dataset/data/test_raw_pdbs/",
            # Antibody-specific parameters
            h_chain=config.get("h_chain", "H"),
            l_chain=config.get("l_chain", "L"),
            modality=modality
        )

        # Create single-item dataset and loader
        dataset = SingleItemDataset(ligand, receptor)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_item_collate_fn) if fabric.global_rank == 0 else None
        loader = fabric.setup_dataloaders(loader, use_distributed_sampler=False) if fabric.global_rank == 0 else None
    else:
        # Use existing .pt files through create_field_loaders
        loader = create_field_loaders(config_nf, split=config["sampling"]["split"], fabric=fabric, sample_points=False, shuffle=False) if fabric.global_rank == 0 else None

    # field_maker
    field_maker, field_maker_receptor = create_field_makers(config, config_nf, fabric)

    # ----------------------
    # start sampling
    fabric.print("start sampling...")

    metrics_sampling = sample(
        loader,
        model,
        enc,
        dec_module,
        config,
        config_nf,
        fabric,
        field_maker,
        field_maker_receptor,
        use_metrics=True,
        num_classes=num_classes,
    )

    if config["wandb"]:
        fabric.log_dict({"sampling": metrics_sampling})


if __name__ == "__main__":
    main()
