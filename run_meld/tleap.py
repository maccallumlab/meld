import subprocess

if __name__ == "__main__":
    # Part 0: Read sequence from a file
    with open("sequence.dat", "r") as file:
        seq = file.read().strip()  # Assuming the sequence is on a single line

    # Part 1: Convert 1-letter amino acid sequence to 3-letter code
    def aa1_to_aa3(seq):
        aa_dict = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }

        converted_sequence = [aa_dict[aa] for aa in seq] if seq else []

        if converted_sequence:
            converted_sequence[0] = 'N' + converted_sequence[0]
            converted_sequence[-1] = 'C' + converted_sequence[-1]

        # Join the sequence with spaces
        final_sequence = " ".join(converted_sequence)
        return final_sequence

    three_code = aa1_to_aa3(seq)
    #print("3-letter code:", aa1_to_aa3(seq))

    # Part 2: Convert to PDB
    def write_tleap(seq, script_name):
        script = f"""
set default PBradii mbondi3
source leaprc.protein.ff14SBonlysc
source leaprc.gaff

mol = sequence {{ {seq} }}
savepdb mol prot.pdb

quit
"""
        with open(script_name, 'w') as f:
            f.write(script)

        subprocess.run(["tleap", "-f", script_name], check=True)

    # Part 3: Prepare system with tleap
    def prep_tleap(pdb, tleap_script):
        with open(tleap_script, 'w') as f:
            f.write(f"""
set default PBradii mbondi3     
source leaprc.protein.ff14SBonlysc
source leaprc.gaff

protein = loadPdb {pdb}
saveAmberParm protein protein.prmtop protein.inpcrd
quit
""")
        subprocess.run(["tleap", "-f", tleap_script], check=True)

    # Part 4: Minimize with sander
    def minimize(min_in, out_prefix):
        with open(min_in, 'w') as f:
            f.write("""
Minimization with implicit solvent
 &cntrl
  imin=1, maxcyc=1000, ncyc=500,
  igb=1, saltcon=0.0,
  cut=999., ntb=0,
 /
""")
        subprocess.run(["sander", "-O", "-i", min_in,
                        "-o", f"{out_prefix}_min.out", "-p", "protein.prmtop",
                        "-c", "protein.inpcrd", "-r", f"{out_prefix}_min.rst"], check=True)

    # Part 5: Convert minimized structure to PDB
    def to_pdb(prmtop, rst, pdb_out):
        subprocess.run(["ambpdb", "-p", prmtop, "-c", rst], stdout=open(pdb_out, 'w'), check=True)

    # Setup and run all
    pdb = "prot.pdb"  # Name from Part 2
    tleap_pdb = "pdb.in"
    tleap_script = "prep.in"
    min_in = "min.in"
    out_prefix = "protein"

    write_tleap(three_code, tleap_pdb)
    prep_tleap(pdb, tleap_script)
    minimize(min_in, out_prefix)
    to_pdb("protein.prmtop", f"{out_prefix}_min.rst", f"{out_prefix}_min.pdb")

