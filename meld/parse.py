#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from collections import namedtuple
from meld.system.restraints import RestraintGroup, TorsionRestraint
from meld.system.restraints import DistanceRestraint, RdcRestraint
from meld.system.restraints import ConstantRamp


aa_map = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIE',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR'
}


# copy the canonical forms
allowed_residues = [aa for aa in aa_map.values()]
# add the alternate protonation states
allowed_residues += ['ASH', 'GLH', 'HIE', 'HID', 'HIP', 'LYN', 'ACE',
                     'OHE', 'NME', 'NHE']


def get_sequence_from_AA1(filename=None, contents=None, file=None,
                          capped=False, nter=None, cter=None):
    """
    Get the sequence from a list of 1-letter amino acid codes.

    :param filename: string of filename to open
    :param contents: string containing contents
    :param file: a file-like object to read from
    :return: a string that can be used to initialize a system
    :raise: RuntimeError on bad input
    :capped: will know that there are caps. Specify which in nter and cter
    :nter: Specify capping residue at the N terminus if not specified
           in sequence
    :cter: Specify capping residue at the C terminus if not specified
           in sequence

    Note: specify exactly one of filename, contents, file
    Note: will have to set options in setup script to skip cmap assignment
    """
    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    sequence = ''.join(lines)

    output = []
    counter = 0
    for aa in sequence:
        try:
            four_letter = aa_map[aa]
            output.append(four_letter)
        except KeyError:
            raise RuntimeError('Unknown amino acid "{}".'.format(aa))
        if counter % 200 == 0 and counter != 0: # then the sequence is getting long
          output.append('\n') # append a newline to keep it from getting too long
          
        counter += 1

    # append terminal qualifiers
    if not capped:
        output[0] = 'N' + output[0]
        output[-1] = 'C' + output[-1]
    else:
        if nter:
            output.insert(0, nter)
        if cter:
            output.append(cter)

    return ' '.join(output)


def get_sequence_from_AA3(filename=None, contents=None, file=None,
                          capped=False, nter=None, cter=None):
    """
    Get the sequence from a list of 3-letter amino acid codes.

    :param filename: string of filename to open
    :param contents: string containing contents
    :param file: a file-like object to read from
    :return: a string that can be used to initialize a system
    :raise: RuntimeError on bad input
    :capped: will know that there are caps. Either read from sequence or
             specified in nter and cter
    :nter: Specify capping residue at the N terminus if not specified
           in sequence
    :cter: Specify capping residue at the C terminus if not specified
           in sequence

    Note: specify exactly one of filename, contents, file
    """
    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    sequence = ' '.join(lines).split()

    output = []
    for aa in sequence:
        if aa not in allowed_residues:
            raise RuntimeError('Unknown residue {}.'.format(aa))
        else:
            output.append(aa)

    # append terminal qualifiers
    if not capped:
        output[0] = 'N' + output[0]
        output[-1] = 'C' + output[-1]
    else:
        if nter:
            output.insert(0, nter)
        if cter:
            output.append(cter)

    return ' '.join(output)


def get_secondary_structure_restraints(system, scaler,
                                       ramp=None,
                                       torsion_force_constant=2.48,
                                       distance_force_constant=2.48,
                                       quadratic_cut=2.0,
                                       filename=None,
                                       contents=None,
                                       file=None):
    """
    Get a list of secondary structure restraints.

    :param system: a System object
    :param scaler: a force scaler
    :param torsion_force_constant: force constant for torsions,
                                   in kJ/mol/(10 degree)^2
    :param distance_force_constant: force constant for distances,
                                    in kJ/mol/Angstrom^2
    :param quadratic_cut: switch from quadratic to linear beyond this
                          distance, Angstrom
    :param filename: string of filename to open
    :param contents: string of contents to process
    :param file: file-like object to read from
    :return: a list of RestraintGroups

    Note: specify exactly one of filename, contents, file.
    """
    if ramp is None:
        ramp = ConstantRamp()

    contents = _get_secondary_sequence(filename, contents, file)
    torsion_force_constant /= 100.
    distance_force_constant *= 100.
    quadratic_cut *= 10.

    groups = []

    helices = _extract_secondary_runs(contents, 'H', 5, 4)
    for helix in helices:
        rests = []
        for index in range(helix.start + 1, helix.end - 1):
            phi = TorsionRestraint(system, scaler, ramp, index, 'C', index+1,
                                   'N', index+1, 'CA', index+1, 'C',
                                   -62.5, 17.5, torsion_force_constant)
            psi = TorsionRestraint(system, scaler, ramp, index+1, 'N', index+1,
                                   'CA', index+1, 'C', index+2, 'N',
                                   -42.5, 17.5, torsion_force_constant)
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(system, scaler, ramp, helix.start+1, 'CA',
                               helix.start+4, 'CA', 0, 0.485, 0.561,
                               0.561 + quadratic_cut, distance_force_constant)
        d2 = DistanceRestraint(system, scaler, ramp, helix.start+2, 'CA',
                               helix.start+5, 'CA', 0, 0.485, 0.561,
                               0.561 + quadratic_cut, distance_force_constant)
        d3 = DistanceRestraint(system, scaler, ramp, helix.start+1, 'CA',
                               helix.start+5, 'CA', 0, 0.581, 0.684,
                               0.684 + quadratic_cut, distance_force_constant)
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = RestraintGroup(rests, len(rests))
        groups.append(group)

    extended = _extract_secondary_runs(contents, 'E', 5, 4)
    for ext in extended:
        rests = []
        for index in range(ext.start + 1, ext.end - 1):
            phi = TorsionRestraint(system, scaler, ramp, index, 'C', index+1,
                                   'N', index+1, 'CA', index+1, 'C',
                                   -117.5, 27.5, torsion_force_constant)
            psi = TorsionRestraint(system, scaler, ramp, index+1, 'N', index+1,
                                   'CA', index+1, 'C', index+2, 'N',
                                   145, 25.0, torsion_force_constant)
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(system, scaler, ramp, ext.start+1, 'CA',
                               ext.start+4, 'CA', 0, 0.785, 1.063,
                               1.063 + quadratic_cut, distance_force_constant)
        d2 = DistanceRestraint(system, scaler, ramp, ext.start+2, 'CA',
                               ext.start+5, 'CA', 0, 0.785, 1.063,
                               1.063 + quadratic_cut, distance_force_constant)
        d3 = DistanceRestraint(system, scaler, ramp, ext.start+1, 'CA',
                               ext.start+5, 'CA', 0, 1.086, 1.394,
                               1.394 + quadratic_cut, distance_force_constant)
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = RestraintGroup(rests, len(rests))
        groups.append(group)

    return groups


def _get_secondary_sequence(filename=None, contents=None, file=None):
    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    sequence = ''.join(lines)
    for ss in sequence:
        if ss not in 'HE.':
            raise RuntimeError(
                'Unknown secondary structure type "{}"'.format(ss))
    return sequence


SecondaryRun = namedtuple('SecondaryRun', 'start end')


def _extract_secondary_runs(content, ss_type, run_length, at_least):
    # mark the elements that have the correct type
    has_correct_type = [1 if ss == ss_type else 0 for ss in content]

    # compute the number of correct elements in each segment
    length = len(content)
    totals = [0] * (length - run_length + 1)
    for index in range(length - run_length + 1):
        totals[index] = sum(has_correct_type[index:index+run_length])

    # add a result whenever we've had at_least correct
    results = []
    for index in range(len(totals)):
        if totals[index] >= at_least:
            results.append(SecondaryRun(index, index+run_length))

    return results


def _handle_arguments(filename, contents, file):
    set_args = [arg for arg in [filename, contents, file] if arg is not None]
    if len(set_args) != 1:
        raise RuntimeError(
            'Must set exactly one of filename, contents or file.')

    if filename:
        return open(filename).read()
    elif contents:
        return contents
    elif file:
        return file.read()


def get_rdc_restraints(system, scaler, ramp=None, filename=None,
                       contents=None, file=None):
    if ramp is None:
        ramp = ConstantRamp()

    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]

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
        kappa = float(cols[7])
        force_const = float(cols[8])
        weight = float(cols[9])

        rest = RdcRestraint(system, scaler, ramp, res_i, atom_i, res_j, atom_j,
                            kappa, obs, tolerance, force_const, weight, expt)
        restraints.append(rest)
    return restraints
