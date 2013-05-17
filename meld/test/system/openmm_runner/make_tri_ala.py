#!/usr/bin/env python
# encoding: utf-8

from meld import system


def main():
    # create the system
    p = system.ProteinMoleculeFromSequence('NALA ALA CALA')
    b = system.SystemBuilder()
    s = b.build_system_from_molecules([p])
    with open('tri_ala.top', 'w') as outfile:
        outfile.write(s._top_string)


if __name__ == '__main__':
    main()
