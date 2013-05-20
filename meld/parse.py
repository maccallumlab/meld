from collections import namedtuple
from meld.system.restraints import RestraintGroup, TorsionRestraint, DistanceRestraint


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
allowed_residues += ['ASH', 'GLH', 'HIE', 'HID', 'HIP', 'LYN']


def get_sequence_from_AA1(filename=None, contents=None, file=None):
    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    sequence = ''.join(lines)

    output = []
    for aa in sequence:
        try:
            four_letter = aa_map[aa]
            output.append(four_letter)
        except KeyError:
            raise RuntimeError('Unknown amino acid "{}".'.format(aa))

    # append terminal qualifiers
    output[0] = 'N' + output[0]
    output[-1] = 'C' + output[-1]
    return ' '.join(output)


def get_sequence_from_AA3(filename=None, contents=None, file=None):
    contents = _handle_arguments(filename, contents, file)
    lines = contents.splitlines()
    lines = [line.strip() for line in lines if not line.startswith('#')]
    items = [line.split() for line in lines]
    sequence = ' '.join(lines).split()

    output = []
    for aa in sequence:
        if not aa in allowed_residues:
            raise RuntimeError('Unknown residue {}.'.format(aa))
        else:
            output.append(aa)
    output[0] = 'N' + output[0]
    output[-1] = 'C' + output[-1]
    return ' '.join(output)


def get_secondary_structure_restraint_groups(system, scaler, force_constant=2.48, filename=None, contents=None, file=None):
    contents = _get_secondary_sequence(filename, contents, file)

    groups = []

    helices = _extract_secondary_runs(contents, 'H', 5, 4)
    for helix in helices:
        rests = []
        for index in range(helix.start + 1, helix.end - 1):
            phi = TorsionRestraint(system, scaler, index, 'C', index+1, 'N', index+1, 'CA',
                                   index+1, 'C', -60, 10, force_constant)
            psi = TorsionRestraint(system, scaler, index+1, 'N', index+1, 'CA', index+1, 'C',
                                   index+2, 'N', -60, 10, force_constant)
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(system, scaler, helix.start+1, 'CA', helix.start+4, 'CA',
                               0, 0.485, 0.561, 999., force_constant)
        d2 = DistanceRestraint(system, scaler, helix.start+2, 'CA', helix.start+5, 'CA',
                               0, 0.485, 0.561, 999., force_constant)
        d3 = DistanceRestraint(system, scaler, helix.start+1, 'CA', helix.start+5, 'CA',
                               0, 0.581, 0.684, 999., force_constant)
        rests.append(d1)
        rests.append(d2)
        rests.append(d3)
        group = RestraintGroup(rests, len(rests))
        groups.append(group)

    extended = _extract_secondary_runs(contents, 'E', 5, 4)
    for ext in extended:
        rests = []
        for index in range(ext.start + 1, ext.end - 1):
            phi = TorsionRestraint(system, scaler, index, 'C', index+1, 'N', index+1, 'CA',
                                   index+1, 'C', -60, 10, force_constant)
            psi = TorsionRestraint(system, scaler, index+1, 'N', index+1, 'CA', index+1, 'C',
                                   index+2, 'N', -60, 10, force_constant)
            rests.append(phi)
            rests.append(psi)
        d1 = DistanceRestraint(system, scaler, helix.start+1, 'CA', helix.start+4, 'CA',
                               0, 0.785, 1.063, 999., force_constant)
        d2 = DistanceRestraint(system, scaler, helix.start+2, 'CA', helix.start+5, 'CA',
                               0, 0.785, 1.063, 999., force_constant)
        d3 = DistanceRestraint(system, scaler, helix.start+1, 'CA', helix.start+5, 'CA',
                               0, 1.086, 1.394, 999., force_constant)
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
        if not ss in 'HE.':
            raise RuntimeError('Unknown secondary structure type "{}"'.format(aa))
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
    set_args = [arg for arg in [filename, contents, file] if not arg is None]
    if len(set_args) != 1:
        raise RuntimeError('Must set exactly one of filename, contents or file.')

    if filename:
        return open(filename).read()
    elif contents:
        return contents
    elif file:
        return file.read()


