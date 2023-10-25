#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Functions to read in sequences, secondary structures, and RDCs

"""

from collections import namedtuple
from typing import List, NewType, Optional, TextIO

from openmm import unit as u  # type: ignore

from meld import interfaces
from meld.system import indexing, restraints, scalers

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
    Get the sequence from 1-letter amino acid codes.

    Args:
        filename: filename to open
        content: contents
        file: object to read from
        capped: Will know that there are caps. Specify which in nter and cter
        nter: specify capping residue at the N terminus if not specified
           in sequence
        cter: specify capping residue at the C terminus if not specified
           in sequence

    Returns:
        string representation that can be used to create a :class:`SubSystem`

    Note:
       Specify exactly one of filename, contents, file
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
    Get the sequence from a 3-letter amino acid codes.

    Args:
        filename: filename to open
        content: contents
        file: object to read from
        capped: Will know that there are caps. Specify which in nter and cter
        nter: specify capping residue at the N terminus if not specified
           in sequence
        cter: specify capping residue at the C terminus if not specified
           in sequence

    Returns:
        string representation that can be used to create a :class:`SubSystem`

    Note:
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
    system: interfaces.ISystem,
    scaler: scalers.RestraintScaler,
    ramp: Optional[scalers.TimeRamp] = None,
    torsion_force_constant: Optional[u.Quantity] = None,
    distance_force_constant: Optional[u.Quantity] = None,
    quadratic_cut: Optional[u.Quantity] = None,
    first_residue: Optional[indexing.ResidueIndex] = None,
    min_secondary_match: int = 4,
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
) -> List[restraints.RestraintGroup]:
    """
    Get a list of secondary structure restraints.

    Args:
        system: the system
        scaler: the force scaler
        ramp: the ramp, default is ConstantRamp
        torsion_force_constant: force constant for torsions, default 2.5e-2 kJ/mol/deg^2
        distance_force_constant: force constant for distances, default 2.5 kJ/mol/nm^2
        quadratic_cut: switch from quadratic to linear beyond this distance, default 0.2 nm
        min_secondary_match: minimum number of elements to match in secondary structure
        first_residue: residue at which these secondary structure restraints start
        filename: filename to open
        content: contents to process
        file: object to read from

    Returns
        A list of :class:`RestraintGroups`

    Note:
       Specify exactly one of filename, contents, file.
    """

    torsion_force_constant = (
        2.5e-2 * u.kilojoule_per_mole / u.degree**2
        if torsion_force_constant is None
        else torsion_force_constant
    )
    distance_force_constant = (
        2500 * u.kilojoule_per_mole / u.nanometer**2
        if distance_force_constant is None
        else distance_force_constant
    )
    quadratic_cut = 0.2 * u.nanometer if quadratic_cut is None else quadratic_cut

    if ramp is None:
        ramp = restraints.ConstantRamp()

    if first_residue is None:
        first_residue = indexing.ResidueIndex(0)
    else:
        assert isinstance(first_residue, indexing.ResidueIndex)

    if min_secondary_match > 5:
        raise RuntimeError(
            "Minimum number of elements to match in secondary structure "
            "must be less than or equal to 5."
        )
    min_secondary_match = int(min_secondary_match)

    contents = _get_secondary_sequence(filename, content, file)

    groups = []

    helices = _extract_secondary_runs(
        contents, "H", 5, min_secondary_match, first_residue
    )
    for helix in helices:
        rests: List[restraints.SelectableRestraint] = []
        for index in range(helix.start + 1, helix.end - 1):
            phi = restraints.TorsionRestraint(
                system,
                scaler,
                ramp,
                system.index.atom(index - 1, "C"),
                system.index.atom(index, "N"),
                system.index.atom(index, "CA"),
                system.index.atom(index, "C"),
                -62.5 * u.degree,
                17.5 * u.degree,
                torsion_force_constant,
            )
            psi = restraints.TorsionRestraint(
                system,
                scaler,
                ramp,
                system.index.atom(index, "N"),
                system.index.atom(index, "CA"),
                system.index.atom(index, "C"),
                system.index.atom(index + 1, "N"),
                -42.5 * u.degree,
                17.5 * u.degree,
                torsion_force_constant,
            )
            rests.append(phi)
            rests.append(psi)
        d1 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(helix.start, "CA"),
            system.index.atom(helix.start + 3, "CA"),
            0 * u.nanometer,
            0.485 * u.nanometer,
            0.561 * u.nanometer,
            0.561 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        d2 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(helix.start + 1, "CA"),
            system.index.atom(helix.start + 4, "CA"),
            0 * u.nanometer,
            0.485 * u.nanometer,
            0.561 * u.nanometer,
            0.561 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        d3 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(helix.start, "CA"),
            system.index.atom(helix.start + 4, "CA"),
            0 * u.nanometer,
            0.581 * u.nanometer,
            0.684 * u.nanometer,
            0.684 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = restraints.RestraintGroup(rests, len(rests))
        groups.append(group)

    extended = _extract_secondary_runs(
        contents, "E", 5, min_secondary_match, first_residue
    )
    for ext in extended:
        rests = []
        for index in range(ext.start + 1, ext.end - 1):
            phi = restraints.TorsionRestraint(
                system,
                scaler,
                ramp,
                system.index.atom(index - 1, "C"),
                system.index.atom(index, "N"),
                system.index.atom(index, "CA"),
                system.index.atom(index, "C"),
                -117.5 * u.degree,
                27.5 * u.degree,
                torsion_force_constant,
            )
            psi = restraints.TorsionRestraint(
                system,
                scaler,
                ramp,
                system.index.atom(index, "N"),
                system.index.atom(index, "CA"),
                system.index.atom(index, "C"),
                system.index.atom(index + 1, "N"),
                145 * u.degree,
                25.0 * u.degree,
                torsion_force_constant,
            )
            rests.append(phi)
            rests.append(psi)
        d1 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(ext.start, "CA"),
            system.index.atom(ext.start + 3, "CA"),
            0 * u.nanometer,
            0.785 * u.nanometer,
            1.063 * u.nanometer,
            1.063 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        d2 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(ext.start + 1, "CA"),
            system.index.atom(ext.start + 4, "CA"),
            0 * u.nanometer,
            0.785 * u.nanometer,
            1.063 * u.nanometer,
            1.063 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        d3 = restraints.DistanceRestraint(
            system,
            scaler,
            ramp,
            system.index.atom(ext.start, "CA"),
            system.index.atom(ext.start + 4, "CA"),
            0 * u.nanometer,
            1.086 * u.nanometer,
            1.394 * u.nanometer,
            1.394 * u.nanometer + quadratic_cut,
            distance_force_constant,
        )
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = restraints.RestraintGroup(rests, len(rests))
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


_SecondaryRun = namedtuple("_SecondaryRun", "start end")


def _extract_secondary_runs(
    content: str, ss_type: str, run_length: int, at_least: int, first_residue: int
) -> List[_SecondaryRun]:
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
            results.append(_SecondaryRun(index, index + run_length))

    # At this point, the runs are zero-based relative to the start of the
    # secondary structure string. Now, we'll add the offset to make them
    # relative to the first residue
    results = [
        _SecondaryRun(s.start + first_residue, s.end + first_residue) for s in results
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
    system: interfaces.ISystem,
    alignment_index: int,
    scaler: restraints.RestraintScaler,
    ramp: Optional[restraints.TimeRamp] = None,
    quadratic_cut: Optional[u.Quantity] = None,
    filename: Optional[str] = None,
    content: Optional[str] = None,
    file: Optional[TextIO] = None,
) -> List[restraints.RdcRestraint]:
    """
    Reads restraints from file and returns as RdcRestraint object.

    Args:
        system: the system object for the restraints to be added to.
        alignment_index: index of the alignment tensor to use
        scaler: object to scale the force constant.
        ramp: ramp, default is ConstantRamp()
        quadratic_cut: restraints become linear beyond this deviation, default 999 Hz
        filename : filename to open
        content : contents to process
        file : object to read from

    Returns:
        list of restraints from input

    Note:
        The value of kappa is assumed to be in units of Hz :math:`Angstrom^3`.

    Note:
        All indexing in the input file is assumed to be 1-based.

    Note:
        The expected order of columns in the input file is:

        - residue i
        - atom name i
        - residue j
        - atom name j
        - observed splitting (:math:`Hz`)
        - experiment (ignored)
        - tolerance (:math:`Hz`)
        - kappa (:math:`Hz \\AA^3`)
        - force constant (:math:`kJ mol^{-1} Hz^{-2}`)
        - weight

    """
    quadratic_cut = 999.0 / u.seconds if quadratic_cut is None else quadratic_cut

    if ramp is None:
        ramp = restraints.ConstantRamp()

    n_align = system.num_alignments
    if alignment_index >= n_align:
        raise ValueError(
            f"Alignment index {alignment_index} out of range for system with {n_align} alignments"
        )

    contents = _handle_arguments(filename, content, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith("#")]

    restraint_list = []
    for line in lines:
        cols = line.split()
        res_i = int(cols[0])
        atom_i = cols[1]
        res_j = int(cols[2])
        atom_j = cols[3]
        obs = float(cols[4])
        expt = int(cols[5])
        tolerance = float(cols[6])
        kappa = float(cols[7])
        force_const = float(cols[8])
        weight = float(cols[9])

        atom_index_i = system.index.atom(res_i, atom_i, one_based=True)
        atom_index_j = system.index.atom(res_j, atom_j, one_based=True)

        rest = restraints.RdcRestraint(
            system,
            scaler,
            ramp,
            atom_index_i,
            atom_index_j,
            kappa * u.second**-1 * u.angstrom**3,
            obs * u.second**-1,
            tolerance * u.second**-1,
            force_const * u.kilojoule_per_mole * u.second**2,
            quadratic_cut,
            weight,
            alignment_index,
        )
        restraint_list.append(rest)
    return restraint_list
