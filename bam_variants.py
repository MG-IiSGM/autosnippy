#!/usr/bin/env python

import os
import logging
import argparse
import subprocess
import shutil
from Bio import SeqIO
import pandas as pd


from misc import check_file_exists, check_create_dir, execute_subprocess, check_remove_file, \
    longest_common_suffix

logger = logging.getLogger()


"""
=============================================================
HEADER
=============================================================

INSTITUTION:IiSGM
AUTHOR: Pedro J. Sola (pedroscampoy@gmail.com)
VERSION=0.1
CREATED: 24 SEp 2020
REVISION:

TODO
    -Explore samtools markdup

================================================================
END_OF_HEADER
================================================================
"""


# COLORS AND AND FORMATTING
"""
http://ozzmaker.com/add-colour-to-text-in-python/
The above ANSI escape code will set the text colour to bright green. The format is;
\033[  Escape code, this is always the same
1 = Style, 1 for normal.
32 = Text colour, 32 for bright green.
40m = Background colour, 40 is for black.
"""
END_FORMATTING = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
YELLOW = '\033[93m'
DIM = '\033[2m'


def run_snippy(r1, r2, reference, output_dir, sample, threads=16, minqual=20, minfrac=0.1, mincov=1):
    """
    snippy --cpus 16 --outdir mysnps --ref Listeria.gbk --R1 FDA_R1.fastq.gz --R2 FDA_R2.fastq.gz
    """
    prefix = os.path.join(output_dir, sample)

    cmd = ["snippy", "--cpus", str(threads), "--outdir", prefix, "--minqual", str(minqual), "--mincov", str(
        mincov), "--minfrac", str(minfrac), "--ref", reference, "--R1", r1, "--R2", r2]

    execute_subprocess(cmd)


def run_snippy_core(input_dir, output_dir, reference, filter_sample = []):
    samples_snippy = []

    output_dir = output_dir + "/core"

    for root, dirs, files in os.walk(input_dir):
        for name in dirs:
            if root == input_dir and not name in filter_sample:
                foldername = os.path.join(root, name)
                samples_snippy.append(foldername)
            elif root == input_dir and name in filter_sample:
                logger.debug(name + " discarded from core FAULTY")

    cmd = ["snippy-core", "-p", output_dir,
           "--ref", reference] + samples_snippy

    execute_subprocess(cmd)


def extract_indels(input_vcf):
    input_vcf = os.path.abspath(input_vcf)
    vcf_dir = ('/').join(input_vcf.split('/')[0:-1])
    output_indel_vcf = os.path.join(vcf_dir, 'snps.indel.vcf')
    with open(output_indel_vcf, 'w+') as fout:
        with open(input_vcf, 'r') as f:
            for line in f:
                if "TYPE=ins" in line or "TYPE=del" in line:
                    fout.write(line)


def merge_vcf(snp_vcf, indel_vcf):
    snp_vcf = os.path.abspath(snp_vcf)
    indel_vcf = os.path.abspath(indel_vcf)

    vcf_dir = ('/').join(snp_vcf.split('/')[0:-1])

    output_complete_vcf = os.path.join(vcf_dir, 'snps.all.vcf')
    with open(output_complete_vcf, 'w+') as fout:
        with open(snp_vcf, 'r') as f1:
            for line in f1:
                fout.write(line)
        with open(indel_vcf, 'r') as f2:
            for line in f2:
                if not line.startswith("#"):
                    fout.write(line)


def create_bamstat(input_bam, output_dir, sample, threads=8):
    output_file = os.path.join(output_dir, sample + ".bamstats")
    cmd = "samtools flagstat --threads {} {} > {}".format(
        str(threads), input_bam, output_file)
    execute_subprocess(cmd, isShell=True)


def create_coverage(input_bam, output_dir, sample):
    output_file = os.path.join(output_dir, sample + ".cov")
    cmd = "samtools depth -aa {} > {}".format(input_bam, output_file)
    # cmd = "bedtools genomecov -ibam {} -d -split > {}".format(input_bam, output_file)
    execute_subprocess(cmd, isShell=True)


def samtools_markdup(args):
    # http://www.htslib.org/doc/samtools.html
    # Add ms and MC tags for markdup to use later
    # samtools fixmate -m namesort.bam fixmate.bam
    # Markdup needs position order
    # samtools sort -o positionsort.bam fixmate.bam
    # Finally mark duplicates
    # samtools markdup positionsort.bam markdup.bam
    pass


def picard_markdup(input_bam):
    # java -jar picard.jar MarkDuplicates \
    #  I=input.bam O=marked_duplicates.bam M=marked_dup_metrics.txt
    #picard_jar = get_picard_path()

    input_bam = os.path.abspath(input_bam)
    #in_param = "I=" + input_bam

    path_file_name = input_bam.split(".")[0]
    file_name = input_bam.split("/")[-1]
    output_markdup = path_file_name + ".rg.markdup.bam"
    output_markdup_sorted = path_file_name + ".rg.markdup.sorted.bam"

    output_dir = ('/').join(input_bam.split('/')[0:-1])
    stat_output_dir = os.path.join(output_dir, "Stats")
    stat_output_file = file_name + ".markdup.metrics.txt"
    stat_output_full = os.path.join(stat_output_dir, stat_output_file)

    check_create_dir(stat_output_dir)

    cmd_markdup = ["picard", "MarkDuplicates", "-I", input_bam,
                   "-O", output_markdup, "-M", stat_output_full]
    execute_subprocess(cmd_markdup)

    # samtools sort: samtools sort $output_dir/$sample".sorted.bam" -o $output_dir/$sample".sorted.bam"
    cmd_sort = ["samtools", "sort", output_markdup,
                "-o", output_markdup_sorted]
    execute_subprocess(cmd_sort)

    # Handled in Haplotype Caller function
    # samtools index: samtools index $output_dir/$sample".sorted.bam"
    # subprocess.run(["samtools", "index", output_markdup_sorted],
    # stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    check_remove_file(input_bam)
    check_remove_file(output_markdup)


def picard_dictionary(args):
    # java -jar picard.jar CreateSequenceDictionary\
    # R=reference.fasta O=reference.dict
    #picard_jar = get_picard_path()

    input_reference = os.path.abspath(args.reference)
    ref_param = "R=" + input_reference

    path_file_list = input_reference.split(".")[:-1]
    path_file_name = ".".join(path_file_list)
    dict_file_name = path_file_name + ".dict"
    out_param = "O=" + dict_file_name

    if os.path.exists(dict_file_name):
        logger.info(dict_file_name + " already EXIST")
    else:
        cmd = ["picard", "CreateSequenceDictionary",
               ref_param, out_param]
        execute_subprocess(cmd)


def samtools_faidx(args):
    # samtools faidx reference.fa

    input_reference = os.path.abspath(args.reference)
    fai_file_name = input_reference + ".fai"

    if os.path.exists(fai_file_name):
        logger.info(fai_file_name + " already EXIST")
    else:
        cmd = ["samtools", "faidx", input_reference]
        execute_subprocess(cmd)


def ivar_trim(input_bam, primers_file, sample, min_length=30, min_quality=20, sliding_window_width=4):
    """
    Usage: ivar trim -i <input.bam> -b <primers.bed> -p <prefix> [-m <min-length>] [-q <min-quality>] [-s <sliding-window-width>]
        Input Options    Description
           -i    (Required) Sorted bam file, with aligned reads, to trim primers and quality
           -b    (Required) BED file with primer sequences and positions
           -m    Minimum length of read to retain after trimming (Default: 30)
           -q    Minimum quality threshold for sliding window to pass (Default: 20)
           -s    Width of sliding window (Default: 4)
           -e    Include reads with no primers. By default, reads with no primers are excluded
        Output Options   Description
           -p    (Required) Prefix for the output BAM file
    """

    input_bam = os.path.abspath(input_bam)
    input_bai = input_bam + ".bai"
    primers_file = os.path.abspath(primers_file)

    prefix = input_bam.split('.')[0] + ".rg.markdup.trimmed"
    output_trimmed_bam = prefix + ".bam"
    output_trimmed_sorted_bam = input_bam.split(
        '.')[0] + ".rg.markdup.trimmed.sorted.bam"

    cmd = ["ivar", "trim", "-i", input_bam, "-b", primers_file, "-p", prefix, "-m",
           str(min_length), "-q", str(min_quality), "-s", str(sliding_window_width), "-e"]
    execute_subprocess(cmd)

    check_remove_file(input_bam)

    cmd_sort = ["samtools", "sort", output_trimmed_bam,
                "-o", output_trimmed_sorted_bam]
    execute_subprocess(cmd_sort)

    check_remove_file(output_trimmed_bam)

    cmd_index = ["samtools", "index", output_trimmed_sorted_bam]
    execute_subprocess(cmd_index)

    check_remove_file(input_bai)


def ivar_variants(reference, input_bam, output_variant, sample, annotation, min_quality=20, min_frequency_threshold=0.8, min_depth=20):
    """
    Usage: samtools mpileup -aa -A -d 0 -B -Q 0 --reference [<reference-fasta] <input.bam> | ivar variants -p <prefix> [-q <min-quality>] [-t <min-frequency-threshold>] [-m <minimum depth>] [-r <reference-fasta>] [-g GFF file]
        Note : samtools mpileup output must be piped into ivar variants
        Input Options    Description
           -q    Minimum quality score threshold to count base (Default: 20)
           -t    Minimum frequency threshold(0 - 1) to call variants (Default: 0.03)
           -m    Minimum read depth to call variants (Default: 0)
           -r    Reference file used for alignment. This is used to translate the nucleotide sequences and identify intra host single nucleotide variants
           -g    A GFF file in the GFF3 format can be supplied to specify coordinates of open reading frames (ORFs). In absence of GFF file, amino acid translation will not be done.
        Output Options   Description
           -p    (Required) Prefix for the output tsv variant file
    """
    ivar_folder = os.path.join(output_variant, 'ivar_raw')
    check_create_dir(ivar_folder)
    prefix = ivar_folder + '/' + sample

    input = {'reference': reference,
             'input_bam': input_bam,
             'prefix': prefix,
             'min_quality': str(min_quality),
             'min_frequency_threshold': str(min_frequency_threshold),
             'min_depth': str(min_depth),
             'annotation': annotation}

    cmd = "samtools mpileup -aa -A -d 0 -B -Q 0 --reference {reference} {input_bam} | \
        ivar variants -p {prefix} -q {min_quality} -t {min_frequency_threshold} -m {min_depth} -r {reference} -g {annotation}".format(**input)

    execute_subprocess(cmd, isShell=True)


def ivar_consensus(input_bam, output_consensus, sample, min_quality=20, min_frequency_threshold=0.8, min_depth=20, uncovered_character='N'):
    """
    ivar consensus
        Usage: samtools mpileup -aa -A -d 0 -Q 0 <input.bam> | ivar consensus -p <prefix> 
        Note : samtools mpileup output must be piped into ivar consensus
        Input Options    Description
           -q    Minimum quality score threshold to count base (Default: 20)
           -t    Minimum frequency threshold(0 - 1) to call consensus. (Default: 0)
                 Frequently used thresholds | Description
                 ---------------------------|------------
                                          0 | Majority or most common base
                                        0.2 | Bases that make up atleast 20% of the depth at a position
                                        0.5 | Strict or bases that make up atleast 50% of the depth at a position
                                        0.9 | Strict or bases that make up atleast 90% of the depth at a position
                                          1 | Identical or bases that make up 100% of the depth at a position. Will have highest ambiguities
           -m    Minimum depth to call consensus(Default: 10)
           -k    If '-k' flag is added, regions with depth less than minimum depth will not be added to the consensus sequence. Using '-k' will override any option specified using -n 
           -n    (N/-) Character to print in regions with less than minimum coverage(Default: N)
        Output Options   Description
           -p    (Required) Prefix for the output fasta file and quality file
    """

    prefix = output_consensus + '/' + sample

    input = {'input_bam': input_bam,
             'prefix': prefix,
             'min_quality': str(min_quality),
             'min_frequency_threshold': str(min_frequency_threshold),
             'min_depth': str(min_depth),
             'uncovered_character': uncovered_character}

    cmd = "samtools mpileup -aa -A -d 0 -B -Q 0  {input_bam} | \
        ivar consensus -p {prefix} -q {min_quality} -t {min_frequency_threshold} -m {min_depth} -n {uncovered_character}".format(**input)

    execute_subprocess(cmd, isShell=True)


def replace_consensus_header(input_fasta):
    with open(input_fasta, 'r+') as f:
        content = f.read()
        header = content.split('\n')[0].strip('>')
        new_header = header.split('_')[1].strip()
        content = content.replace(header, new_header)
        f.seek(0)
        f.write(content)
        f.truncate()


if __name__ == '__main__':
    logger.info("#################### BAM RECALL #########################")
