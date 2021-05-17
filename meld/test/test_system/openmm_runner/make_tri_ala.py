#!/usr/bin/env python
# encoding: utf-8

#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import system


def main():
    # create the system
    p = system.SubSystemFromSequence("NALA ALA CALA")
    b = system.SystemBuilder()
    s = b.build_system([p])
    with open("tri_ala.top", "w") as outfile:
        outfile.write(s._top_string)


if __name__ == "__main__":
    main()
