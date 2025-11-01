import torch

from pyuul import VolumeMaker


class Voxelizer(torch.nn.Module):
    """
    Voxelizer module for converting molecular structures to voxel representations.

    Args:
        grid_dim (int): The dimension of the voxel grid (default: 64).
        resolution (float): The resolution of the voxel grid (default: 0.25).
        radius (float): The radius used for voxelization (default: 0.5).
        cubes_around (int): The number of cubes around each atom used for voxelization (default: 8).
        device (str): The device to use for computation (default: "cuda").

    Attributes:
        grid_dim (int): The dimension of the voxel grid.
        device (str): The device used for computation.
        radius (float): The radius used for voxelization.
        resolution (float): The resolution of the voxel grid.
        cubes_around (int): The number of cubes around each atom used for voxelization.
        vol_maker (VolumeMaker.Voxels): The voxelization module.

    """

    def __init__(
        self,
        grid_dim: int = 64,
        resolution: float = 0.25,
        radius: float = 0.5,
        cubes_around: int = 8,
        device="cuda"
    ):
        super(Voxelizer, self).__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.radius = radius
        self.resolution = resolution
        self.cubes_around = cubes_around

        self.vol_maker = VolumeMaker.Voxels(
            device=device,
            sparse=False,
        )

    def forward(self, batch: list, num_channels: int = 7, dummy=25) -> torch.Tensor:
        """
        Forward pass of the Voxelizer module.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """
        return self.mol2vox(batch, num_channels=num_channels, dummy=dummy)

    def mol2vox(self, batch: list, num_channels: int = 7, dummy=25) -> torch.Tensor:
        """
        Convert a batch of molecular structures to voxel representations.

        Args:
            batch (list): The input batch of molecular structures.
            num_channels (int): The number of channels in the voxel grid (default: 7).

        Returns:
            torch.Tensor: The voxelized representation of the input batch.

        """
        batch['coords'] = batch['coords'].to(self.device)
        batch['atoms_channel'] = batch['atoms_channel'].to(self.device)
        batch['radius'] = batch['radius'].to(self.device)

        # dumb coordinate to center ligand and receptor voxel
        batch = self._add_dumb_coords(batch, dummy=dummy)

        # voxelize
        voxels = []
        batch_sz = batch["coords"].shape[0]
        n_chuncks = 4 if batch_sz > 16 else 1
        chk = batch["coords"].shape[0] // n_chuncks
        for i in range(n_chuncks):
            voxels_ = self.vol_maker(
                batch["coords"][i * chk:(i + 1) * chk],
                batch["radius"][i * chk:(i + 1) * chk],
                batch["atoms_channel"][i * chk:(i + 1) * chk],
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=num_channels,
            )
            # extract center box (and get rid of dumb coordinates)
            c = voxels_.shape[-1] // 2
            box_min, box_max = c - self.grid_dim // 2, c + self.grid_dim // 2
            voxels_ = voxels_[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
            voxels.append(voxels_)
        voxels = torch.cat(voxels, axis=0)

        return voxels

    def _add_dumb_coords(self, batch: dict, dummy=25) -> dict:
        """
        Add dumb coordinates to the input batch for centering the ligand and receptor voxel.

        Args:
            batch (dict): The input batch of molecular structures.

        Returns:
            dict: The modified batch with dumb coordinates.

        """
        bsz = batch['coords'].shape[0]
        return {
            "coords": torch.cat(
                (batch['coords'], torch.Tensor(bsz, 1, 3).fill_(-dummy).to(self.device), torch.Tensor(bsz, 1, 3).fill_(dummy).to(self.device)), 1
            ),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.Tensor(bsz, 2).fill_(0).to(self.device)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.Tensor(bsz, 2).fill_(.5).to(self.device), ), 1
            )
        }
