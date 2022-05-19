#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Methods to read template coordinates from a system object and build elastic network restraints prior to simulation for a specific chain.
"""

import numpy as np  # type: ignore
from scipy.spatial.distance import pdist  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore
from typing import List, Optional, Union

from meld.system import restraints
from meld.system import meld_system
from meld.system import scalers
from meld import unit as u

def create_elastic_network_restraints(
    system: meld_system.System,
    scaler: scalers.RestraintScaler,
    chain_index: int,
    cutoff: float = 1.0,
    restrained_atom: str = 'CA',
    residue_separation: int = 3,
    k: u.Quantity = 2500 * u.kilojoule_per_mole / u.nanometer ** 2,
    write_file: Optional[str] = None,

):
    """
    Create elastic network restraints between specified backbone atoms within a single chain
    """
    # Get all residues in the chain
    chain = list(system.topology.chains())[chain_index]
    residues = []
    for residue in chain.residues():
        residues.append(residue)

    # Extract atom indices for CA atoms
    ca_indices = []
    for residue in residues:
        # Not using chainid because that converts to relative numbering where as openmm uses absolute numbering within chains
        ca_index = system.index.atom(residue.index, restrained_atom, expected_resname=residue.name,)
        ca_indices.append(ca_index)
        
    # Get initial coordinates from the system
    coordinates = system.template_coordinates
    ca_coordinates = coordinates[ca_indices]
    
    # Calculate pairwise distances
    dists = pdist(ca_coordinates)
    
    # Transform dists into a nres by nres distance matrix
    dist_map = squareform(dists)
    
    # Grab upper (or lower doesn't matter chose upper arbitrarily) diagonal because dist_map is symmetric
    dist_map = np.triu(dist_map) 
    
    # Make sure the length of the dist_map is the same as the length of residues
    # This will not be the case if restrained_atom isn't present in every residue
    # In that case, this function won't work
    assert dist_map.shape[0] == len(residues)
    
    # Triu sets off diagonal elements to zero so we look for things above zero and below our cutoff
    close_pairs = np.argwhere((dist_map > 0) & (dist_map < cutoff))
    
    # Collect the precise distance of each close pair
    close_distances = []
    for pair in close_pairs:
        close_distance = dist_map[pair[0]][pair[1]]
        close_distances.append(close_distance)
    
    # Create restraints
    rests = []
    for pair, dist in zip(close_pairs, close_distances):
        i, j = pair[0], pair[1]
        
        # Only create restraints if residues are separated by at least residue_separation
        if abs(i - j) >= residue_separation:
            rest = system.restraints.create_restraint(
                "distance",
                scaler=scaler,
                atom1 = system.index.atom(residues[i].index, restrained_atom, expected_resname=residues[i].name,), 
                atom2 = system.index.atom(residues[j].index, restrained_atom, expected_resname=residues[j].name,),
                r1 = 0.0 * u.nanometer,
                r2 = 0.0 * u.nanometer,
                r3 = dist * u.nanometer,
                r4 = (dist + 0.2) * u.nanometer,
                k = k,
                )
            rests.append(rest)
            
    # If write_file is set - write the restraints to a separate file.
    # Mostly a sanity check to compare to previous ways to generate restraints
    if write_file:
        with open(write_file, 'w') as f:
            for pair, dist in zip(close_pairs, close_distances):
                i, j = pair[0], pair[1]
                if abs(i - j) >= residue_separation:
                    f.write(f"{i} {restrained_atom} {residues[i].name} {j} {restrained_atom} {residues[j].name} {dist} {k} \n")
            
                
    return rests

def add_elastic_network_restraints(
    system: meld_system.System,
    rests: List[restraints.SelectableRestraint],
    active_fraction: float = 1.0,
    max_grp_len: int = 64,
):
    """
    For performance reasons, we add restraints in groups of grp_len
    """
    collection: List[Union[restraints.RestraintGroup, restraints.SelectableRestraint]]  = []
    grp : List[restraints.SelectableRestraint] = []
    for rest in rests:
        # Check to see if the length of the group is equal to the max group length
        if len(grp) == max_grp_len:
            # If so add group to collection and empty group
            g = system.restraints.create_restraint_group(grp, len(grp)) 
            collection.append(g)
            grp = []
        
        # Append restraints to group until max_grp_len is reached
        grp.append(rest)
    
    
    # When out of the for loop there will be remaining restraints that have a total length less than max_grp_len
    # So we add the remaining restraints to the collection
    g = system.restraints.create_restraint_group(grp, len(grp)) 
    collection.append(g)
    
    # Add restraints to system
    system.restraints.add_selectively_active_collection(collection, int(len(collection) * active_fraction))
    
    
    
    
    
    
    
    
    
    
    
    



