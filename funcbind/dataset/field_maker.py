import math
import numpy as np
import torch
from torch import nn
from funcbind.utils.constants import PADDING_INDEX

class FieldMaker(nn.Module):
    def __init__(
        self,
        config: dict,
        cubes_around: int = 5,
        sample_points: bool = True,
        sample_grid: bool = True,
    ):
        super(FieldMaker, self).__init__()
        self.grid_dim_low_res = config["dset"]["latent_grid_dim"]
        self.resolution_low_res = config["dset"]["latent_resolution"]
        self.grid_dim = config["dset"]["grid_dim"]
        self.resolution = config["dset"]["resolution"]
        if "latent_ligand_radius" in config["dset"] and config["dset"]["latent_ligand_radius"] is not None:
            self.radius_scale = np.sqrt(config["dset"]["latent_ligand_radius"])
        else:
            self.radius_scale = np.sqrt(config["dset"]["grid_dim"] / config["dset"]["latent_grid_dim"])
        self.cubes_around = cubes_around
        self.vol_maker = Voxels()
        self.elements = config["dset"]["elements"]
        self.laplace = False
        self.dumb = config["dset"].get("dumb", 10.0)  # default value for dumb coordinates
        self.sample_points = sample_points
        self.sample_grid = sample_grid

    @torch.no_grad()
    def forward(self, batch: dict, num_channels: int)-> tuple:
        # avoid using forward, and use compute_occupancies and/or compute_voxel_grid directly
        occs_points = None
        if self.sample_points:
            occs_points = self.compute_occupancies(batch, num_channels=num_channels)

        occs_grid = None
        if self.sample_grid:
            occs_grid = self.compute_voxel_grid(batch, num_channels)

        return occs_points, occs_grid

    @torch.no_grad()
    def compute_voxel_grid(self, batch, num_channels=None):
        batch = self._add_dumb_coords(batch)
        occs_grid = self.vol_maker(batch["coords"], batch["radius"] * self.radius_scale, batch["atoms_channel"], resolution=self.resolution_low_res, cubes_around_atoms_dim=self.cubes_around, numberchannels=len(self.elements) if num_channels is None else num_channels)
        # get center box (remove dumb coordinates)
        center = (occs_grid.shape[-1]) // 2 - 1
        box_min, box_max = center - self.grid_dim_low_res // 2, center + self.grid_dim_low_res // 2
        assert box_min >= 0 and box_max <= occs_grid.shape[-1], "Box size exceeds grid dimensions"
        occs_grid = occs_grid[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
        if not occs_grid.is_contiguous():  # making it compatible with torch.compile
            occs_grid = occs_grid.contiguous()
        return occs_grid


    @torch.no_grad()
    def compute_occupancies(self, mols, num_channels=None) -> torch.Tensor:
        # scale the xs to be in Angstrom
        xs = mols["xs"] * (self.resolution * self.grid_dim / 2)
        occs = torch.zeros(xs.size(0), xs.size(1), len(self.elements) if num_channels is None else num_channels, device=xs.device)
        unique_channels = mols["atoms_channel"].int().unique()
        valid_channels = unique_channels[unique_channels != torch.tensor(PADDING_INDEX, dtype=xs.dtype)]

        # Expand dimensions for broadcasting
        coords = mols["coords"].unsqueeze(1).repeat(1, len(valid_channels), 1, 1)
        # mask the coords with PADDING_INDEX
        mask = mols["atoms_channel"].unsqueeze(1) == valid_channels.unsqueeze(0).unsqueeze(2)
        coords[~mask] = torch.tensor(PADDING_INDEX, dtype=xs.dtype)

        # Compute distance between x and each atom and the exponent
        exponent = torch.cdist(xs.unsqueeze(1), coords, p=1 if self.laplace else 2)
        radius = mols["radius"]
        radius[radius == PADDING_INDEX] = 1.0
        exponent /= (radius.unsqueeze(1).unsqueeze(2) * 0.93)
        if not self.laplace:
            exponent = exponent ** 2
        # Mask to avoid computation when exponent > 10
        valid_mask = exponent <= 10
        exponent = torch.clamp(exponent, max=10)

        # Compute occupancy for the channels
        exp_term = torch.exp(-exponent)
        log_term = torch.log(1 - exp_term)
        log_term[~valid_mask] = 0  # Set log_term to 0 where exponent > 10 to avoid unnecessary computation

        occs[:, :, valid_channels] = (1 - torch.exp(log_term.sum(3).transpose(1, 2)))
        if not occs.is_contiguous():  # making it compatible with torch.compile
            occs = occs.contiguous()
        return occs

    @torch.no_grad()
    def _add_dumb_coords(self, batch: dict) -> dict:
        """
        Add dumb coordinates to center the molecule.

        Args:
            batch (dict): A dictionary containing the molecular data.

        Returns:
            dict: A dictionary containing the molecular data with dumb coordinates added.
        """
        bsz = batch['coords'].shape[0]
        dumb_coord = batch['coords'][batch['coords'] != torch.tensor(PADDING_INDEX, dtype=batch['coords'].dtype)].abs().max() + self.dumb
        dumb_coord_repeat = dumb_coord.repeat(bsz, 1, 3)
        return {
            "coords": torch.cat((batch['coords'], -dumb_coord_repeat, dumb_coord_repeat), 1),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.full((bsz, 2), 0, dtype=batch['atoms_channel'].dtype, device=batch['atoms_channel'].device)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.full((bsz, 2), .5, dtype=batch['radius'].dtype, device=batch['radius'].device), ), 1
            )
        }

    def set_sample_points(self, sample_points: bool):
        self.sample_points = sample_points


# This is highly inspired by the code from pyuul
class Voxels(torch.nn.Module):
    def __init__(self):
        """
        Initializes the Voxels class.

        This constructor calls the parent class's constructor and initializes
        the `boxsize` attribute to None.
        """
        super(Voxels, self).__init__()
        self.boxsize = None

    def _transform_coordinates(self,coords,radius=None):
        """
        Transforms the given coordinates by applying dilatation and translation.

        Parameters:
        coords (numpy.ndarray): The coordinates to be transformed.
        radius (float, optional): An optional radius to be transformed. Defaults to None.

        Returns:
        tuple: A tuple containing the transformed coordinates and, if provided, the transformed radius.
        """
        coords = (coords*self.dilatation) - self.translation
        if radius is not None:
            radius = radius*self.dilatation
            return coords,radius
        else:
            return coords

    def _define_spatial_conformation(self,mincoords,cubes_around_atoms_dim,resolution):
        self.translation = (mincoords-(cubes_around_atoms_dim)).unsqueeze(1)
        self.dilatation = 1.0 / resolution

    def forward(self, coords, radius, channels, numberchannels=None, resolution=1, cubes_around_atoms_dim=5):
        channels_int = channels.to(torch.int16)
        padding_mask = channels_int.ne(PADDING_INDEX)
        if numberchannels is None:
            if padding_mask.any():
                numberchannels = int(channels_int[padding_mask].max().cpu().data + 1)
            else:
                numberchannels = 0
        self.featureVectorSize = numberchannels

        arange_type = torch.int16

        gx = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        gy = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        gz = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        self.lato = gx.shape[0]

        x1 = gx.unsqueeze(1).expand(self.lato, self.lato).unsqueeze(-1)
        x2 = gy.unsqueeze(0).expand(self.lato, self.lato).unsqueeze(-1)

        xy = torch.cat([x1, x2], dim=-1).unsqueeze(2).expand(self.lato, self.lato, self.lato, 2)
        x3 = gz.unsqueeze(0).unsqueeze(1).expand(self.lato, self.lato, self.lato).unsqueeze(-1)

        del gx, gy, gz, x1, x2

        self.standard_cube = torch.cat([xy, x3], dim=-1).unsqueeze(0).unsqueeze(0)

        mincoords = torch.min(coords[:, :, :], dim=1)[0]
        mincoords = torch.trunc(mincoords / resolution)
        box_size_x = (math.ceil(torch.max(coords[padding_mask][:,0])/resolution)-mincoords[:,0].min())+(2*cubes_around_atoms_dim+1)
        box_size_y = (math.ceil(torch.max(coords[padding_mask][:,1])/resolution)-mincoords[:,1].min())+(2*cubes_around_atoms_dim+1)
        box_size_z = (math.ceil(torch.max(coords[padding_mask][:,2])/resolution)-mincoords[:,2].min())+(2*cubes_around_atoms_dim+1)

        self._define_spatial_conformation(mincoords,cubes_around_atoms_dim,resolution)	#define the spatial transforms to coordinates
        coords,radius = self._transform_coordinates(coords,radius)

        boxsize = (int(box_size_x),int(box_size_y),int(box_size_z))
        self.boxsize=boxsize

        if max(boxsize)<256:
            self.dtype_indices=torch.uint8
        else:
            self.dtype_indices = torch.int16

        # actual calculation
        batch = coords.shape[0]
        L = coords.shape[1]

        discrete_coordinates = torch.trunc(coords.data).to(self.dtype_indices)

        # implicit_cube_formation
        radius = radius.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [96, 157, 1, 1, 1]
        atNameHashing = channels_int.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        coords = coords.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        discrete_coordinates = discrete_coordinates.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [96, 157, 1, 1, 1, 3]
        distmat_standard_cube = torch.norm(
            coords - ((discrete_coordinates + self.standard_cube + 1) + 0.5 * resolution), dim=-1).to(
            coords.dtype)

        atNameHashing = atNameHashing.long()

        exponent = distmat_standard_cube[padding_mask] ** 2 / (.93 ** 2 * radius[padding_mask] ** 2)
        exp_mask = exponent.ge(10)
        exponent = torch.masked_fill(exponent, exp_mask, 10)
        volume_cubes = torch.exp(-exponent)

        batch_list = torch.arange(batch, device=coords.device).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch,L,self.lato,self.lato,self.lato)

        cubes_coords = (discrete_coordinates[padding_mask] + self.standard_cube.squeeze(0) + 1)[~exp_mask]
        atNameHashing = atNameHashing[padding_mask].expand(-1,self.lato,self.lato,self.lato)

        volume = torch.zeros(batch, boxsize[0]+1, boxsize[1]+1, boxsize[2]+1, self.featureVectorSize, device=coords.device, dtype=coords.dtype)
        index = (batch_list[padding_mask][~exp_mask].view(-1).long(), cubes_coords[:,0].long(), cubes_coords[:,1].long(), cubes_coords[:,2].long(), atNameHashing[~exp_mask])
        volume_cubes=volume_cubes[~exp_mask].view(-1)

        volume_cubes = torch.log(1 - volume_cubes.contiguous())
        volume = 1- torch.exp(volume.index_put(index,volume_cubes,accumulate=True))
        volume=volume.permute(0,4,1,2,3)
        return volume
