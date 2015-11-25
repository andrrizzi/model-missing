#!/usr/bin/env python

"""
Script to model missing residues with Modeller. The missing residues are
detected by checking the SEQRES entry of the PDB file. The Python API of
Modeller must be installed.

Examples
--------
Model missing residues of structure 4WIS in the directory "original_pdb/"
and save the result in the directory "modeller_files/", keep hetatoms and
add trailig residues

    python model_missing.py -i original_pdb/4WIS.pdb -o modeller_files/ --hetatm --addtrail

"""

import os
from collections import OrderedDict
from contextlib import contextmanager

import modeller
from modeller import automodel
from modeller import scripts

# Constants and configuration
# ----------------------------

SUFFIX_COMPLETE = '_fill'

# Utility functions
# ----------------------------

@contextmanager
def working_directory(path):
    """Context to set the working directory to the given path."""
    current_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_dir)

residue_map = {'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
               'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
               'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
               'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'}


def get_seq_res(pdb_file_name):
    """Extract the pdb sequence from SEQRES entries.

    Parameters
    ----------
    pdb_file_name : str
        Path to the PDB file to parse.

    Return
    ------
    chain_sequences : OrderedDict of (character, str)
        The keys are chain ids and the values are the sequences.

    """
    chain_sequences = OrderedDict()
    pdb_file = open(pdb_file_name, 'r')
    for line in pdb_file:

        # Residue
        if line[:6] == 'SEQRES':
            chain_id = line[11]
            if chain_id not in chain_sequences:
                chain_sequences[chain_id] = ''
            residues = line[19:].split()
            for res in residues:
                chain_sequences[chain_id] += residue_map[res]

    pdb_file.close()

    return chain_sequences


def extract_pdb_code(pdb_file_name):
    """Extract pdb code from path to pdb file."""
    pdb_code = pdb_file_name.split('.')[-2]  # Remove file extension
    pdb_code = pdb_code.split(os.path.sep)[-1]
    return pdb_code


def format_seq_pir(chain_seq, hetatm_seq):
    """Format the chain sequences to be represented in PIR format.

    Parameters
    ----------
    chain_seq : iterable of str
        List of sequences for each chain.
    hetatm_seq : iterable of str
        List of sequences of hetatm for each chain to append at the end.

    Return
    ------
    formatted_seq : str
        The formatted sequence for PIR format.

    """
    WIDTH = 75

    # Join strings and add end of sequence character
    formatted_seq = '/'.join(chain_seq)
    if len(hetatm_seq) > 0:
        formatted_seq += '/' + '/'.join(hetatm_seq)
    formatted_seq += '*'

    # Set maximum width for sequences to 75 characters
    n_res = WIDTH
    while n_res < len(formatted_seq):
        formatted_seq = formatted_seq[:n_res] + '\n' + formatted_seq[n_res:]
        n_res += WIDTH + 1  # to take into account the '\n' character

    return formatted_seq + '\n'


def create_alignment(env, pdb_file_name, parse_hetatm, add_trailing):
    """Create a PIR file for Modeller to model missing residues.

    The file contains tha alignment between the reference sequence as described
    by the SEQRES field in the PDB file and the actual residues modeled in the
    PDB file.

    Return
    ------
    gap_ranges : list of pair of int
        The intervals (extremes included) of the indices of missing residues.

    """
    pdb_code = extract_pdb_code(pdb_file_name)
    pir_file_name = pdb_code + '.pir'

    if parse_hetatm:
        env.io.hetatm = True

    # Write sequence in Modeller's PIR format based on
    # non-missing residues in the pdb_code.pdb file
    model = modeller.model(env, file=pdb_code)
    aln = modeller.alignment(env)
    aln.append_model(model, align_codes=pdb_code)
    aln.write(file=pir_file_name)

    # Read header from temp file to reproduce in final alignment file
    pir_file = open(pir_file_name, 'r')
    header = pir_file.readlines()[:3]
    pir_file.close()

    # Create a sequence with gaps where residues are missing
    hetatm = OrderedDict()  # number of hetatm for each chain
    chain_gap_ranges = OrderedDict()  # gap ranges for each chain
    chain_seq_missing = OrderedDict()  # sequence for each chain
    for res in model.residues:

        res_num = int(res.num)
        res_chain = res.chain.name

        if res_chain not in chain_seq_missing:
            hetatm[res_chain] = ''
            chain_gap_ranges[res_chain] = []
            chain_seq_missing[res_chain] = ''

        if res.hetatm:
            hetatm[res_chain] += res.code
        else:
            seq_num = len(chain_seq_missing[res_chain]) + 1
            if res_num != seq_num:  # insert gaps
                n_gaps = res_num - seq_num
                chain_gap_ranges[res_chain].append((seq_num - 1, seq_num + n_gaps - 2))
                chain_seq_missing[res_chain] += '-' * n_gaps

            chain_seq_missing[res_chain] += res.code

    # Get complete sequence from SEQRES entries of PDB file
    chain_seq_complete = get_seq_res(pdb_file_name)

    if len(chain_seq_complete) != len(chain_seq_missing):
        raise Exception('Different number of chains from Modeller and PDB SEQRES!')

    # Handle trailing missing residues
    if add_trailing:  # add final trailing residues
        for chain_id in chain_seq_missing:
            if len(chain_seq_complete[chain_id]) > len(chain_seq_missing[chain_id]):
                len_comp = len(chain_seq_complete[chain_id])
                len_miss = len(chain_seq_missing[chain_id])
                n_gaps = len_comp - len_miss
                chain_seq_missing[chain_id] += '-' * n_gaps
                chain_gap_ranges[chain_id].append((len_miss, len_comp - 1))
    else:
        for chain_id in chain_seq_missing:
            # Remove initials
            if chain_seq_missing[chain_id][0] == '-':
                last_gap = chain_gap_ranges[chain_id][0][1]
                chain_gap_ranges[chain_id] = chain_gap_ranges[chain_id][1:]
                chain_seq_missing[chain_id] = chain_seq_missing[chain_id][last_gap+1:]
                chain_seq_complete[chain_id] = chain_seq_complete[chain_id][last_gap+1:]

                for (i, interval) in enumerate(chain_gap_ranges[chain_id]):
                    new_interval = (interval[0] - last_gap - 1, interval[1] - last_gap - 1)
                    chain_gap_ranges[chain_id][i] = new_interval

            # Remove finals in chain_seq_complete
            n_res = len(chain_seq_complete[chain_id]) - len(chain_seq_missing[chain_id])
            chain_seq_complete[chain_id] = chain_seq_complete[chain_id][:-n_res]

    # Check that everything went right
    for chain_id in chain_seq_missing:
        # Check chain lengths
        if len(chain_seq_missing[chain_id]) != len(chain_seq_complete[chain_id]):
            raise Exception('Different chain length from Modeller and PDB SEQRES!')

        # Check residue alignments
        for (res_miss, res_comp) in zip(chain_seq_missing[chain_id],
                                        chain_seq_complete[chain_id]):
            if res_miss != '-' and res_miss != res_comp:
                err_str = 'Residues {:!s} and {:!s} are different in Modeller and PDB SEQRES!'.format(
                    res_miss, res_comp)
                raise Exception(err_str)

        # Check gap range
        for (i, res) in enumerate(chain_seq_missing[chain_id]):
            if res == '-':
                found_interval = False
                for interval in chain_gap_ranges[chain_id]:
                    if interval[0] <= i <= interval[1]:
                        found_interval = True
                if not found_interval:
                    raise Exception('Computed the wrong gap ranges!')

    # Create alignment file that will be used by Modeller
    pir_file = open(pir_file_name, 'w')

    for line in header:
        pir_file.write(line)
    pir_file.write(format_seq_pir(chain_seq_missing.values(), hetatm.values()))

    pir_file.write(header[1].strip() + SUFFIX_COMPLETE + '\n')
    pir_file.write('sequence:pdb_file::::::::\n')
    pir_file.write(format_seq_pir(chain_seq_complete.values(), hetatm.values()))

    pir_file.close()

    # Fix gap ranges taking into account chain lengths
    cumulative_chain_len = [0 for x in chain_seq_missing]
    for (i, seq) in enumerate(chain_seq_missing.values()):
        if i < len(chain_seq_missing) - 1:
            cumulative_chain_len[i + 1] = len(seq) + cumulative_chain_len[i]

    gap_ranges = [(interval[0] + cumulative_chain_len[i], interval[1] + cumulative_chain_len[i])
                  for (i, chain) in enumerate(chain_gap_ranges.values()) for interval in chain]

    return gap_ranges


def model_structure(env, pdb_code, missing_res_ranges, verbose=False):

    if verbose:
        modeller.log.verbose()

    class MyModeller(automodel.loopmodel):
        def select_atoms(self):
            sel_ranges = [self.residue_range(*interval) for interval in missing_res_ranges]
            return modeller.selection(*sel_ranges)

        def select_loop_atoms(self):
            sel_ranges = [self.residue_range(*interval) for interval in missing_res_ranges]
            return modeller.selection(*sel_ranges)

    a = MyModeller(env, alnfile=pdb_code+'.pir',
                   knowns=pdb_code, sequence=pdb_code+SUFFIX_COMPLETE)
    a.starting_model = 1
    a.ending_model = 10

    a.loop.starting_model = 1
    a.loop.ending_model = 1
    a.loop.md_level = automodel.refine.fast

    a.make()


def main(pdb_file_name, out_dir, parse_hetatm=True, add_trailing=False):

    pdb_code = extract_pdb_code(pdb_file_name)
    input_dir = os.path.abspath(os.path.dirname(pdb_file_name))

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create and configure Modeller environment
    env = modeller.environ()
    env.io.atom_files_directory = [input_dir]

    # Create alignment file in out_dir
    with working_directory(out_dir):
        gap_ranges = create_alignment(env, pdb_file_name, parse_hetatm, add_trailing)
        model_structure(env, pdb_code, gap_ranges)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputpdb', metavar='PDBFILE', required=True,
                        help='Path to protein PDB file', dest='pdb_file')
    parser.add_argument('-o', '--outdir', metavar='OUTPUTDIR', required=True,
                        help='Path to output directory where modeller files are saved',
                        dest='out_dir')
    parser.add_argument('--hetatm', action='store_true', dest='hetatm',
                        help='If specified, parses HETATM atoms in PDB file.')
    parser.add_argument('--addtrail', action='store_true', dest='addtrail',
                        help='If specified, add the also the missing trailing residues.')
    args = parser.parse_args()

    pdb_file_name = os.path.abspath(args.pdb_file)
    out_dir = os.path.abspath(args.out_dir)

    main(pdb_file_name, out_dir, parse_hetatm=args.hetatm, add_trailing=args.addtrail)
