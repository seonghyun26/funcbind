from typing import Optional
import pyrosetta
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.core.pose import append_pose_to_pose
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

pyrosetta.init("-corrections:beta_nov16 -flip_HNQ true -no_optH false -mute all")

def interface_energy(pdb_path: str, ag_pdb_path: Optional[str] = None, interface: str = 'HL_A') -> float:
    pose = pose_from_pdb(pdb_path)
    if ag_pdb_path is not None:
        ag_pose = pose_from_pdb(ag_pdb_path)
        append_pose_to_pose(pose, ag_pose)
        pose.conformation().detect_disulfides()
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    try:
        mover.apply(pose)
    except Exception as e:
        return float('nan')
    return pose.scores['dG_separated']
