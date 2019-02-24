#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Functions to read in sequences, secondary structures, and RDCs.

"""

from typing import List, Optional, TextIO, NewType
from collections import namedtuple
from meld.system import System
from meld.system.restraints import (
    RestraintGroup,
    Restraint,
    TorsionRestraint,
    DistanceRestraint,
    RdcRestraint,
    TimeRamp,
    ConstantRamp,
    RestraintScaler,
)

SequenceString = NewType("SequenceString", str)
SecondaryString = NewType("SecondaryString", str)


aa_map = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIE",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


# copy the canonical forms
allowed_residues = [aa for aa in aa_map.values()]
# add the alternate protonation states
allowed_residues += [
    "ASH",
    "GLH",
    "HIE",
    "HID",
    "HIP",
    "LYN",
    "ACE",
    "OHE",
    "NME",
    "NHE",
]


def get_sequence_from_AA1(
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
    capped: bool = False,
    nter: Optional[str] = None,
    cter: Optional[str] = None,
) -> SequenceString:
    """
    Get the sequence from a list of 1-letter amino acid codes.

    Parameters
    ----------
    filename : string
        Filename to open
    content : string
        Contents
    file : file-like
        Object to read from
    capped
        Will know that there are caps. Specify which in nter and cter
    nter
        Specify capping residue at the N terminus if not specified
           in sequence
    cter
        Specify capping residue at the C terminus if not specified
           in sequence

    Returns
    -------
    string
        Used to initialize a system

    Raises
    ------
    RuntimeError
        Error raised on bad input

    Notes
    -----
    Specify exactly one of filename, contents, file

    Will have to set options in setup script to skip cmap assignment

    """
    contents = _handle_arguments(filename, content, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith("#")]
    sequence = "".join(lines)

    output = []
    for aa in sequence:
        try:
            four_letter = aa_map[aa]
            output.append(four_letter)
        except KeyError:
            raise RuntimeError(f'Unknown amino acid "aa".')

    # append terminal qualifiers
    if not capped:
        output[0] = "N" + output[0]
        output[-1] = "C" + output[-1]
    else:
        if nter:
            output.insert(0, nter)
        if cter:
            output.append(cter)

    max_aa_per_line = 100
    groups = [
        output[i : i + max_aa_per_line]
        for i in range(0, len(sequence), max_aa_per_line)
    ]
    lines = [" ".join(group) for group in groups]
    return SequenceString("\n".join(lines))


def get_sequence_from_AA3(
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
    capped: bool = False,
    nter: Optional[str] = None,
    cter: Optional[str] = None,
) -> SequenceString:
    """
    Get the sequence from a list of 3-letter amino acid codes.

    Parameters
    ----------
    filename : string
        Filename to open
    content : string
        Contains contents
    file : file-like object
        Object to read from
    capped
        Will know that there are caps. Specify which in nter and cter
    nter
        Specify capping residue at the N terminus if not specified
           in sequence
    cter
        Specify capping residue at the C terminus if not specified
           in sequence

    Returns
    -------
    string
        Used to initialize a system

    Raises
    ------
    RuntimeError
        Error on based input

    Notes
    -----
    Specify exactly one of filename, contents, file

    """
    contents = _handle_arguments(filename, content, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith("#")]
    sequence = " ".join(lines).split()

    output = []
    for aa in sequence:
        if aa not in allowed_residues:
            raise RuntimeError(f"Unknown residue {aa}.")
        else:
            output.append(aa)

    # append terminal qualifiers
    if not capped:
        output[0] = "N" + output[0]
        output[-1] = "C" + output[-1]
    else:
        if nter:
            output.insert(0, nter)
        if cter:
            output.append(cter)

    return SequenceString(" ".join(output))


def get_secondary_structure_restraints(
    system: System,
    scaler: RestraintScaler,
    ramp: Optional[TimeRamp] = None,
    torsion_force_constant: float = 2.48,
    distance_force_constant: float = 2.48,
    quadratic_cut: float = 2.0,
    first_residue: int = 1,
    min_secondary_match: int = 4,
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
) -> List[RestraintGroup]:
    """
    Get a list of secondary structure restraints.

    Parameters
    ----------
    system : a System object
        The system
    scaler : a force scaler
        The force
    ramp:
        The ramp, default is ConstantRamp
    torsion_force_constant : float
        Force constant for torsions, in kJ/mol/(10 degree)^2
    distance_force_constant : float
        Force constant for distances, in kJ/mol/Angstrom^2
    quadratic_cut : float
        Switch from quadratic to linear beyond this distance, Angstrom
    min_secondary_match : int
        Minimum number of elements to match in secondary structure,
    first_residue : int
        Residue at which to delineate peptide vs. protein,
    filename : string
        Filename to open
    content : string
        Contents to process
    file : file-like object
        Object to read from

    Returns
    -------
    groups : list
        A list of RestraintGroups

    Notes
    -----
    Specify exactly one of filename, contents, file.

    """
    if ramp is None:
        ramp = ConstantRamp()

    if min_secondary_match > 5:
        raise RuntimeError(
            "Minimum number of elements to match in secondary structure "
            "must be less than or equal to 5."
        )
    min_secondary_match = int(min_secondary_match)

    contents = _get_secondary_sequence(filename, content, file)
    torsion_force_constant /= 100.
    distance_force_constant *= 100.
    quadratic_cut *= 10.

    groups = []

    helices = _extract_secondary_runs(
        contents, "H", 5, min_secondary_match, first_residue
    )
    for helix in helices:
        rests: List[Restraint] = []
        for index in range(helix.start + 1, helix.end - 1):
            phi = TorsionRestraint(
                system,
                scaler,
                ramp,
                index - 1,
                "C",
                index,
                "N",
                index,
                "CA",
                index,
                "C",
                -62.5,
                17.5,
                torsion_force_constant,
            )
            psi = TorsionRestraint(
                system,
                scaler,
                ramp,
                index,
                "N",
                index,
                "CA",
                index,
                "C",
                index + 1,
                "N",
                -42.5,
                17.5,
                torsion_force_constant,
            )
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(
            system,
            scaler,
            ramp,
            helix.start,
            "CA",
            helix.start + 3,
            "CA",
            0,
            0.485,
            0.561,
            0.561 + quadratic_cut,
            distance_force_constant,
        )
        d2 = DistanceRestraint(
            system,
            scaler,
            ramp,
            helix.start + 1,
            "CA",
            helix.start + 4,
            "CA",
            0,
            0.485,
            0.561,
            0.561 + quadratic_cut,
            distance_force_constant,
        )
        d3 = DistanceRestraint(
            system,
            scaler,
            ramp,
            helix.start,
            "CA",
            helix.start + 4,
            "CA",
            0,
            0.581,
            0.684,
            0.684 + quadratic_cut,
            distance_force_constant,
        )
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = RestraintGroup(rests, len(rests))
        groups.append(group)

    extended = _extract_secondary_runs(
        contents, "E", 5, min_secondary_match, first_residue
    )
    for ext in extended:
        rests = []
        for index in range(ext.start + 1, ext.end - 1):
            phi = TorsionRestraint(
                system,
                scaler,
                ramp,
                index - 1,
                "C",
                index,
                "N",
                index,
                "CA",
                index,
                "C",
                -117.5,
                27.5,
                torsion_force_constant,
            )
            psi = TorsionRestraint(
                system,
                scaler,
                ramp,
                index,
                "N",
                index,
                "CA",
                index,
                "C",
                index + 1,
                "N",
                145,
                25.0,
                torsion_force_constant,
            )
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(
            system,
            scaler,
            ramp,
            ext.start,
            "CA",
            ext.start + 3,
            "CA",
            0,
            0.785,
            1.063,
            1.063 + quadratic_cut,
            distance_force_constant,
        )
        d2 = DistanceRestraint(
            system,
            scaler,
            ramp,
            ext.start + 1,
            "CA",
            ext.start + 4,
            "CA",
            0,
            0.785,
            1.063,
            1.063 + quadratic_cut,
            distance_force_constant,
        )
        d3 = DistanceRestraint(
            system,
            scaler,
            ramp,
            ext.start,
            "CA",
            ext.start + 4,
            "CA",
            0,
            1.086,
            1.394,
            1.394 + quadratic_cut,
            distance_force_constant,
        )
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = RestraintGroup(rests, len(rests))
        groups.append(group)

    return groups


def _get_secondary_sequence(
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
) -> SecondaryString:
    contents = _handle_arguments(filename, content, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith("#")]
    sequence = "".join(lines)
    for ss in sequence:
        if ss not in "HE.":
            raise RuntimeError(f'Unknown secondary structure type "{ss}"')
    return SecondaryString(sequence)


SecondaryRun = namedtuple("SecondaryRun", "start end")


def _extract_secondary_runs(
    content: str, ss_type: str, run_length: int, at_least: int, first_residue: int
) -> List[SecondaryRun]:
    # mark the elements that have the correct type
    has_correct_type = [1 if ss == ss_type else 0 for ss in content]

    # compute the number of correct elements in each segment
    length = len(content)
    totals = [0] * (length - run_length + 1)
    for index in range(length - run_length + 1):
        totals[index] = sum(has_correct_type[index : index + run_length])

    # add a result whenever we've had at_least correct
    results = []
    for index in range(len(totals)):
        if totals[index] >= at_least:
            results.append(SecondaryRun(index, index + run_length))

    # At this point, the runs are zero-based relative to the start of the
    # secondary structure string. Now, we'll add the offset to make them
    # one-based and relative to the first residue
    results = [
        SecondaryRun(s.start + first_residue, s.end + first_residue) for s in results
    ]

    return results


def _handle_arguments(
    filename: Optional[str], contents: Optional[str], file: Optional[TextIO]
) -> str:
    set_args = [arg for arg in [filename, contents, file] if arg is not None]
    if len(set_args) != 1:
        raise RuntimeError("Must set exactly one of filename, contents or file.")

    if filename:
        content = open(filename).read()
    elif contents:
        content = contents
    elif file:
        content = file.read()
    return content


def get_rdc_restraints(
    system: System,
    scaler: RestraintScaler,
    ramp: Optional[TimeRamp] = None,
    quadratic_cut: float = 99999.,
    scale_factor: float = 1.0e4,
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
) -> List[RdcRestraint]:
    """
    Reads restraints from file and returns as RdcRestraint object.

    Parameters
    ----------
    system : meld.system.System
        The system object for the restraints to be added to.
    scaler : meld.system.restraints.RestraintScaler
        Object to scale the force constant.
    ramp : meld.system.restraints.TimeRamp
        Ramp, default is ConstantRamp()
    quadratic_cut : float
        Restraints become linear beyond this deviation s^-1
    scale_factor: float
        Scale factor for kappa and alignment tensor
    filename : string
        Filename to open
    content : string
        Contents to process
    file : a file_like object
        Object to read from

    Returns
    -------
    restraints: RdcREstraint object
        Restraints from file

    Notes
    -----

    The value of `kappa` will be scaled down by `scale_factor`. This will
    result in the alignment tensor being scaled up by `scale_factor`.
    Ideally, the largest values of the scaled alignment tensor should be
    approximately 1. As typical values of the alignment are on the order
    of 1e-4, the default value of 1e4 is a reasonable guess. The value
    of `scale_factor` must be the same for all experiments that share the
    same alignment.
    """
    if ramp is None:
        ramp = ConstantRamp()

    contents = _handle_arguments(filename, content, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith("#")]

    restraints = []
    for line in lines:
        cols = line.split()
        res_i = int(cols[0])
        atom_i = cols[1]
        res_j = int(cols[2])
        atom_j = cols[3]
        obs = float(cols[4])
        expt = int(cols[5])
        tolerance = float(cols[6])
        kappa = float(cols[7]) / scale_factor
        force_const = float(cols[8])
        weight = float(cols[9])

        rest = RdcRestraint(
            system,
            scaler,
            ramp,
            res_i,
            atom_i,
            res_j,
            atom_j,
            kappa,
            obs,
            tolerance,
            force_const,
            quadratic_cut,
            weight,
            expt,
        )
        restraints.append(rest)
    return restraints
