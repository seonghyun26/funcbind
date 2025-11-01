from dataclasses import dataclass
from typing import Any, Optional

################################################################################
# Chemical elements
@dataclass
class Element:
    symbol: str
    vdw_radius: float
    color_ligand: Optional[Any]
    color_ligand_sparse: Optional[Any]

# The order defines the hashing. Atomic radii from
# https://en.wikipedia.org/wiki/Van_der_Waals_radius
# or https://www.webelements.com/fluorine/atom_sizes.html
ELEMENTS = [
    Element("C", vdw_radius=1.7, color_ligand="Greys", color_ligand_sparse="Greys_r"),
    Element("O", vdw_radius=1.52, color_ligand="Reds", color_ligand_sparse="Reds_r"),
    Element("N", vdw_radius=1.55, color_ligand="Blues", color_ligand_sparse="Blues_r"),
    Element(
        "S",
        vdw_radius=1.8,
        color_ligand=[[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],
        color_ligand_sparse=[[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],
    ),
    Element("F", vdw_radius=1.47, color_ligand="Greens", color_ligand_sparse="Greens_r"),
    Element("Cl", vdw_radius=1.75, color_ligand="Mint", color_ligand_sparse="Mint_r"),
    Element("P", vdw_radius=1.8, color_ligand="Oranges", color_ligand_sparse="Oranges_r"),
    Element("Br", vdw_radius=1.85, color_ligand="Purples", color_ligand_sparse="Purples_r"),
    Element(
        "I",
        vdw_radius=1.98,
        color_ligand=[[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],
        color_ligand_sparse=[[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],
    ),
    Element("B", vdw_radius=1.92, color_ligand=None, color_ligand_sparse=None),
    Element("Mg", vdw_radius=1.73, color_ligand="Magenta", color_ligand_sparse="Magenta_r"),
    Element("Ca", vdw_radius=2.62, color_ligand="Magenta", color_ligand_sparse="Magenta_r"),
    Element("Mn", vdw_radius=2.45, color_ligand="Magenta", color_ligand_sparse="Magenta_r"),
    Element("H", vdw_radius=1.2, color_ligand="Magenta", color_ligand_sparse="Magenta_r"),
]

ELEMENT_2_HASH_FUNCBIND = {e.symbol: i for i, e in enumerate(ELEMENTS)}
ELEMENT_2_VDW_RADIUS = {e.symbol: e.vdw_radius for e in ELEMENTS}


################################################################################
# Crossdocked dataset and general
N_RECEPTOR_ELEMENTS = 4
N_LIGAND_ELEMENTS = 7
PADDING_INDEX = 999

CHANNEL_TO_ATM_NB = {
    0: 6,   # C
    1: 8,   # O
    2: 7,   # N
    3: 16,  # S
    4: 9,   # F
    5: 17,  # Cl
    6: 15,  # P
    7: 35,  # Br
    8: 1,   # H
}

################################################################################
PEPTIDE_ALPHABET = "XACDEFGHIKLMNPQRSTVWYacgut"


################################################################################
# SabDab dataset
# SABDAB chain IDs. Everything >1 is antigen
AB_CHAIN_DICT = {'H': 0, 'L': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'I': 9}


def is_antibody_dataset(config):
    return (config["dset"]["input_dataset"] == "omni_v1" and config["dset"]["use_single_dataset"] == "sabdab")


################################################################################
