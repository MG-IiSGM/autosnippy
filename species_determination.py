#!/usr/bin/env python

import os
import logging
import sys
import argparse
import pandas as pd
import numpy as np
import re
import subprocess
from misc import check_create_dir, extract_sample, execute_subprocess

logger = logging.getLogger()


"""
=============================================================
HEADER
=============================================================

INSTITUTION:IiSGM
AUTHOR: Pedro J. Sola (pedroscampoy@gmail.com)
VERSION=0.1
CREATED: 09 July 2019
REVISION:

================================================================
END_OF_HEADER
================================================================
"""

END_FORMATTING = '\033[0m'
WHITE_BG = '\033[0;30;47m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
BLUE = '\033[34m'
CYAN = '\033[36m'
YELLOW = '\033[93m'
DIM = '\033[2m'


def get_arguments():

    parser = argparse.ArgumentParser(
        prog='snptb.py', description='Pipeline to call variants (SNVs) with any non model organism. Specialised in Mycobacterium Tuberculosis')

    input_group = parser.add_argument_group('Input', 'Input parameters')

    input_group.add_argument('-r', '--r1_file', metavar="reference",
                             type=str, required=True, help='REQUIRED. File to map against')
    input_group.add_argument('-R', '--r2_file', metavar="sample",
                             type=str, required=True, help='Sample to identify further files')

    output_group = parser.add_argument_group(
        'Output', 'Required parameter to output results')

    output_group.add_argument('-o', '--output', type=str, required=True,
                              help='REQUIRED. Output directory to extract all results')

    params_group = parser.add_argument_group(
        'Parameters', 'parameters for diferent stringent conditions')

    params_group.add_argument('-c', '--mincov', type=int, required=False,
                              default=20, help='Minimun coverage to add samples into analysis')
    params_group.add_argument('-T', '--threads', type=str, dest="threads",
                              required=False, default=4, help='Threads to use')
    params_group.add_argument('-M', '--memory', type=str, dest="memory",
                              required=False, default=8, help='MAx memory to use')

    arguments = parser.parse_args()

    return arguments


def refseq_masher(r1_file, r2_file, output_file, threads=16, max_results=50):
    """
    refseq_masher contains --top-n-results 50 -p 16 -o HPR3641322-50.contains2.tsv \
     HPR3641322-50_S27_L000_R1_001.fastq.gz HPR3641322-50_S27_L000_R2_001.fastq.gz
    """
    r1 = os.path.abspath(r1_file)
    r2 = os.path.abspath(r2_file)

    output_file = os.path.abspath(output_file)

    cmd = ["refseq_masher", "contains", "--top-n-results",
           str(max_results), "-p", str(threads), "-o", output_file, r1, r2]

    execute_subprocess(cmd)


def zcat_concat_reads(args):
    """
    decompress gz r1 and r2 into a combined .fastq with the common sample in the same directory
    """

    r1 = os.path.abspath(args.r1_file)
    r2 = os.path.abspath(args.r2_file)

    sample = extract_sample(r1, r2)

    output_dir = ("/").join(r1.split("/")[:-1])
    output_name = sample + ".fastq"
    output_file = os.path.join(output_dir, output_name)

    cmd = ["zcat", r1, r2]
    # execute_subprocess(cmd)
    with open(output_file, "w+") as outfile:
        # calculate coverage and save it in th eoutput file
        subprocess.run(cmd,
                       stdout=outfile, stderr=subprocess.PIPE, check=True, universal_newlines=True)

    return output_file


def mash_screen(r1_file, out_dir, r2_file=False, winner=True, threads=16, mash_database="/home/laura/DATABASES/Mash/bacteria_mash.msh"):
    # https://mash.readthedocs.io/en/latest/index.html
    # https://gembox.cbcb.umd.edu/mash/refseq.genomes.k21s1000.msh #MASH refseq database
    # mash screen -w -p 4 ../refseq.genomes.k21s1000.msh 4_R1.fastq.gz 4_R2.fastq.gz > 4.winner.screen.tab
    # identity, shared-hashes, median-multiplicity, p-value, query-ID, query-comment

    if not os.path.isfile(mash_database):
        logger.info(RED + BOLD + "Mash database can't be found\n" + END_FORMATTING + "You can download it typing:\n\
            wget https://gembox.cbcb.umd.edu/mash/refseq.genomes.k21s1000.msh")
        sys.exit(1)

    r1_file = os.path.abspath(r1_file)

    sample = extract_sample(r1_file, r2_file)

    check_create_dir(out_dir)
    species_output_name = sample + ".screen.tab"
    species_output_file = os.path.join(out_dir, species_output_name)

    cmd = ["mash", "screen", "-p", str(threads), mash_database, r1_file]

    if winner == True:
        cmd.insert(2, "-w")
    # Use both r1 and r2 instead of just r1(faster)
    if r2_file:
        r2_file = os.path.abspath(r2_file)
        cmd.append(r2_file)

    prog = cmd[0]
    param = cmd[1:]

    try:
        # execute_subprocess(cmd)
        with open(species_output_file, "w+") as outfile:
            # calculate mash distance and save it in output file
            command = subprocess.run(cmd,
                                     stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
        if command.returncode == 0:
            logger.info(GREEN + "Program %s successfully executed" %
                        prog + END_FORMATTING)
        else:
            print(RED + BOLD + "Command %s FAILED\n" % prog + END_FORMATTING
                  + BOLD + "WITH PARAMETERS: " +
                  END_FORMATTING + " ".join(param) + "\n"
                  + BOLD + "EXIT-CODE: %d\n" % command.returncode +
                  "ERROR:\n" + END_FORMATTING + command.stderr)
    except OSError as e:
        sys.exit(RED + BOLD + "failed to execute program '%s': %s" % (prog,
                                                                      str(e)) + END_FORMATTING)


def extract_species(row):
    split_row = row['query-comment'].split(" ")
    if split_row[0].startswith("[") and split_row[1].endswith("]"):
        species = (" ").join([split_row[3], split_row[4]])
    else:
        species = (" ").join([split_row[1], split_row[2]])
    return species


def extract_accession(row):
    split_row = row['query-comment'].split(" ")
    if split_row[0].startswith("[") and split_row[1].endswith("]"):
        accession = split_row[2]
    else:
        accession = split_row[0]
    return accession


def import_mash_screen_to_pandas(screen_file):
    dataframe = pd.read_csv(screen_file, sep="\t", names=['identity', 'shared-hashes',
                                                          'median-multiplicity', 'p-value',
                                                          'query-ID', 'query-comment'])

    dataframe['Species'] = dataframe.apply(extract_species, axis=1)
    dataframe['Accession'] = dataframe.apply(extract_accession, axis=1)
    dataframe['GCF'] = dataframe['query-ID'].str.split(
        "_").str[0:2].str.join('_')
    dataframe['ASM'] = dataframe['query-ID'].str.split("_").str[2]
    dataframe['Hash_1'] = dataframe['shared-hashes'].str.split("/").str[0]
    dataframe['Hash_2'] = dataframe['shared-hashes'].str.split("/").str[1]

    to_int = ['Hash_1', 'Hash_2']

    for column in dataframe.columns:
        if column in to_int:
            dataframe[column] = dataframe[column].astype(int)

    dataframe['Hash_fr'] = dataframe['Hash_1'] / dataframe['Hash_2']

    return dataframe


def extract_species_from_screen(screen_file, identity_threshold=0.9):

    screen_dataframe = import_mash_screen_to_pandas(screen_file)

    df_index = screen_dataframe[screen_dataframe.identity > identity_threshold]
    # max_hash = df_index['Hash_fr'].max()
    hash_values = df_index.Hash_fr.values.tolist()
    hash_values.sort(reverse=True)

    main_species = df_index['Species'][df_index['Hash_fr']
                                       == hash_values[0]].values[0]

    species_report = "Main species: " + "<i>" + main_species + "</i>" + "<br />"

    # <p style="padding-right: 5px;">My Text Here</p>.
    if len(hash_values) > 1:
        for index, hash_value in enumerate(hash_values):
            species_hashed = df_index['Species'][df_index['Hash_fr']
                                                 == hash_value].values[0]
            if index != 0:
                if hash_value > 0.6:
                    species_line = "Another high represented species found: " + \
                        "<i>" + species_hashed + "</i>" + "<br />"
                elif hash_value < 0.3:
                    species_line = "Another less represented species found: " + \
                        "<i>" + species_hashed + "</i>" + "<br />"
                else:
                    species_line = "Another mild represented species found: " + \
                        "<i>" + species_hashed + "</i>" + "<br />"

                species_report = species_report + species_line

    return main_species, species_report


def kraken(r1_file, r2_file, report, kraken2_db, krona_html, threads=34):

    cmd_kraken = "kraken2 --db {} --use-names --threads {} --report {} --gzip-compressed {} {} --output -".format(
        kraken2_db, str(threads), report, r1_file, r2_file)

    # print(cmd_kraken)
    execute_subprocess(cmd_kraken, isShell=True)

    cmd_krona = "ktImportTaxonomy -m 3 -q 2 -t 5 {} -o {}".format(
        report, krona_html)

    # print(cmd_krona)
    execute_subprocess(cmd_krona, isShell=True)


if __name__ == '__main__':
    logger.info("#################### SPECIES #########################")
    args = get_arguments()
    # zcat_concat_reads(args)

    # mash_screen(args, winner=True, r2=False,
    #            mash_database="/home/laura/DATABASES/Mash/refseq.genomes.k21s1000.msh")
