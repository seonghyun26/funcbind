import os

import numpy as np
import plotly.graph_objects as go
import torch

from funcbind.utils.constants import ELEMENTS
from funcbind.utils.utils_base import makedir, supress_stdout

COLORS_LIGAND = [e.color_ligand for e in ELEMENTS]
COLORS_LIGAND_SPARSE = [
    [[0, "black"], [1, "black"]],           # Channel 1: Black
    [[0, "darkblue"], [1, "lightblue"]],    # Channel 2: Blue (reversed)
    [[0, "darkred"], [1, "lightcoral"]],    # Channel 3: Red (reversed)
    [[0, "gold"], [1, "lightyellow"]],      # Channel 4: Yellow (reversed)
    [[0, "darkorange"], [1, "moccasin"]],   # Channel 5: Orange (reversed)
    [[0, "darkgreen"], [1, "lightgreen"]],  # Channel 6: Green (reversed)
    [[0, "darkviolet"], [1, "plum"]],       # Channel 7: Purple (reversed)
    [[0, "saddlebrown"], [1, "tan"]],       # Channel 8: Brown (reversed)
] + [e.color_ligand_sparse for e in ELEMENTS[8:]]  # Keep original colors for channels beyond 8
COLORS_POCKET = [[(0, "gray"), (1, "rgb(0.5,0.5,0.5)")] for _ in ELEMENTS]
COLORS_POCKET_SPARSE = ["greys" for _ in ELEMENTS]


def visualize_ligand_receptor(
    ligand: torch.Tensor,
    receptor: torch.Tensor,
    name: str = "temp",
    dirname: str = "figures/",
    threshold=0.7,
    downsample: int = 1,
    to_png: bool = True,
    to_html: bool = False,
    sparse: bool = True,
    peaks: bool = False,
):
    """
    Visualizes the ligand and receptor volumes (voxel grids) using a 3D plot.

    Args:
        ligand (torch.Tensor): The ligand volume tensor.
        receptor (torch.Tensor): The receptor volume tensor.
        name (str, optional): The name of the output file. Defaults to "temp".
        dirname (str, optional): The directory to save the output files. Defaults to "figures/".
        threshold (float, optional): The threshold value for voxel visualization. Defaults to 0.1.
        downsample (int, optional): The downsampling factor for voxel visualization. Defaults to 1.
        to_png (bool, optional): Whether to save the visualization as a PNG image. Defaults to True.
        to_html (bool, optional): Whether to save the visualization as an HTML file. Defaults to False.
    """
    if ligand is not None:
        assert len(ligand.shape) == 4
        ligand = ligand.cpu()
    if receptor is not None:
        assert len(receptor.shape) == 4
        receptor = receptor.cpu()

    makedir(dirname)
    fig = go.Figure()

    for voxel, is_receptor in [[receptor, True], [ligand, False]]:
        if voxel is None:
            continue
        if downsample > 1:
            voxel = torch.nn.functional.avg_pool3d(voxel, (downsample, downsample, downsample))
        else:
            voxel = voxel.numpy()
        if sparse:
            colors = COLORS_POCKET_SPARSE if is_receptor else COLORS_LIGAND_SPARSE
        else:
            colors = COLORS_POCKET if is_receptor else COLORS_LIGAND
        # voxel = voxel.squeeze()
        voxel[voxel < threshold] = 0
        if not sparse:
            X, Y, Z = np.mgrid[: voxel.shape[-3], : voxel.shape[-2], : voxel.shape[-1]]

        for channel in range(voxel.shape[0]):
            if not sparse:
                voxel_channel = voxel[channel : channel + 1]
                if voxel_channel.sum().item() == 0:
                    continue
                fig.add_volume(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=voxel_channel.flatten(),
                    isomin=0.19,
                    isomax=0.2,
                    opacity=0.1,
                    surface_count=17,
                    colorscale=colors[channel],
                    showscale=False,
                )
            else:
                voxel_channel = voxel[channel]
                if voxel_channel.sum().item() == 0:
                    continue
                non_zero_indices = np.nonzero(voxel_channel)

                # Special handling for peaks
                if is_receptor and peaks:
                    # Make peaks yellow and bigger
                    fig.add_trace(
                        go.Scatter3d(
                            x=non_zero_indices[0],
                            y=non_zero_indices[1],
                            z=non_zero_indices[2],
                            mode="markers",
                            marker=dict(
                                size=8,  # Bigger size for peaks
                                color="yellow",  # Yellow color for peaks
                                opacity=1.0,
                                symbol="diamond",  # Different shape for peaks
                            ),
                            name=f"Peak_Ch{channel}",
                            showlegend=True,
                        )
                    )
                else:
                    # Regular visualization
                    fig.add_trace(
                        go.Scatter3d(
                            x=non_zero_indices[0],
                            y=non_zero_indices[1],
                            z=non_zero_indices[2],
                            mode="markers",
                            marker=dict(
                                size=1 if not (is_receptor and peaks) else 2,
                                color=voxel_channel[non_zero_indices],
                                colorscale=colors[channel],
                                opacity=0.3 if is_receptor and peaks else 1.0,
                            ),
                            name=f"{'Receptor' if is_receptor else 'Ligand'}_Ch{channel}",
                            showlegend=peaks,
                        )
                    )
        if sparse:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, voxel.shape[-3]]),
                    yaxis=dict(range=[0, voxel.shape[-2]]),
                    zaxis=dict(range=[0, voxel.shape[-1]]),
                    aspectmode="cube",
                ),
            )
    if to_html:
        fig.write_html(f"{dirname}/{name}.html")
    if to_png:
        try:
            fig.write_image(f"{dirname}/{name}.png")
        except Exception as e:
            print("Failed to save the figure as a PNG image:", e)


def remove_bond_from_pdb(path_sdf, fname, fname_temp):
    cmd = f'grep -v "^CONECT" {path_sdf}/{fname_temp} > {path_sdf}/{fname}'
    os.system(cmd)
    os.system(f"rm {path_sdf}/{fname_temp}")


def openbabel_sdf_to_pdb(path_sdf, fname, fname_sdf=None):
    if fname_sdf is not None:
        cmd = f"obabel {path_sdf}/{fname_sdf} -opdb -O {path_sdf}/{fname} --title  end > /dev/null 2>&1"
    else:
        cmd = f"obabel {path_sdf}/*sdf -opdb -O {path_sdf}/{fname} --title  end > /dev/null 2>&1"
    os.system(cmd)


@supress_stdout
def convert_sdf_to_pdb(path_sdf: str, fname: str = None, fabric = None, fname_sdf = None, verbose=False) -> None:
    """
    Convert multiple .sdf files to a single .pdb file.

    Args:
        path_sdf (str): The path to the directory containing the .sdf file.
        fname (str, optional): The name of the output .pdb file. If not provided, the default name "sample.pdb" will be used.

    Returns:
        None
    """
    fname = "samples.pdb" if fname is None else fname
    first_part = fname.split(".")[0]
    if verbose:
        fabric.print(f">> process .sdf files and save in .pdb in {path_sdf}")

    # Convert to temp .pdb file
    openbabel_sdf_to_pdb(path_sdf, fname=f"{first_part}_temp.pdb", fname_sdf=fname_sdf)

    # Convert to final .pdb without bond information
    remove_bond_from_pdb(path_sdf, fname, fname_temp=f"{first_part}_temp.pdb")
