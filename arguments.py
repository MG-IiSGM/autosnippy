#!/usr/bin/env python

import os
import argparse

# ARGUMENTS


def get_arguments():

    parser = argparse.ArgumentParser(
        prog='autosnippy.py', description='Pipeline to call variants (SNVs) with any non model haploid organism using snippy')

    input_group = parser.add_argument_group('Input', 'Input parameters')

    input_group.add_argument('-i', '--input', dest="input_dir", metavar="input_directory",
                             type=str, required=True, help='REQUIRED.Input directory containing all fast[aq] files')
    input_group.add_argument('-r', '--reference', metavar="reference",
                             type=str, required=True, help='REQUIRED. File to map against')
    input_group.add_argument('-s', '--sample', metavar="sample", type=str,
                             required=False, help='Sample to identify further files')
    input_group.add_argument('-L', '--sample_list', type=str, required=False,
                             help='Sample names to analyse only in the file supplied')

    quality_group = parser.add_argument_group(
        'Quality parameters', 'parameters for diferent triming conditions')

    quality_group.add_argument('-c', '--coverage20', type=int, default=70, required=False,
                               help='Minimum percentage of coverage at 20x to clasify as uncovered (Default 70)')

    quality_group.add_argument('-u', '--unmapped', type=int, default=25, required=False,
                               help='Maximum percentage of unmapped reads to classify as uncovered Default: 25')

    quality_group.add_argument('-n', '--min_snp', type=int, required=False,
                               default=30, help='SNP number to pass quality threshold')

    output_group = parser.add_argument_group(
        'Output', 'Required parameter to output results')

    output_group.add_argument('-o', '--output', type=str, required=True,
                              help='REQUIRED. Output directory to extract all results')
    output_group.add_argument('-C', '--noclean', required=False,
                              action='store_false', help='Clean unwanted files for standard execution')

    params_group = parser.add_argument_group(
        'Parameters', 'parameters for diferent stringent conditions')

    params_group.add_argument('-T', '--threads', type=str, dest="threads",
                              required=False, default=32, help='Threads to use')
    params_group.add_argument('-M', '--memory', type=str, dest="memory",
                              required=False, default=32, help='Max memory to use')

    annot_group = parser.add_argument_group(
        'Annotation', 'parameters for variant annotation')

    annot_group.add_argument('-B', '--annot_bed', type=str, default=[],
                             required=False, action='append', help='BED file to annotate')
    annot_group.add_argument('-V', '--annot_vcf', type=str, default=[],
                             required=False, action='append', help='VCF file to annotate')
    annot_group.add_argument('-F', '--annot_fasta', type=str, default=[],
                             required=False, action='append', help='FASTA file to annotate')
    annot_group.add_argument('-A', '--annot_aa', type=str, default=[],
                             required=False, action='append', help='aminoacid file to annotate')
    annot_group.add_argument('-R', '--remove_bed', type=str, default=False,
                             required=False, help='BED file with positions to remove')
    annot_group.add_argument('-P', '--extract_bed', required=False, type=str,
                             default=False, help='BED file with important positions or genes to annotate')
    annot_group.add_argument('--mash_database', dest="mash_db", type=str, required=False,
                             default=False, help='MASH ncbi annotation containing species database')
    annot_group.add_argument("--kraken2", dest="kraken2_db", type=str, default=False,
                             required=False, help="Kraken2 database")
    annot_group.add_argument('--snpeff_database', type=str, required=False,
                             default=False, help='snpEFF annotation database')

    compare_group = parser.add_argument_group(
        'Compare', 'parameters for compare_snp')

    compare_group.add_argument('-S', '--only_snp', required=False,
                               action='store_true', help='Create the results only with SNPs, removing INDELs')
    compare_group.add_argument('-w', '--window', required=False,
                               type=int, default=2, help='Number of snps in 10 to discard: default 2')
    compare_group.add_argument('--core', required=False,
                               action='store_true', help='Run snippy-core')
    compare_group.add_argument('--min_threshold_discard_uncov_sample', required=False,
                               type=float, default=0.5, help='min_threshold_discard_uncov_sample')
    compare_group.add_argument('--min_threshold_discard_uncov_pos', required=False,
                               type=float, default=0.5, help='min_threshold_discard_uncov_pos')
    compare_group.add_argument('--min_threshold_discard_htz_sample', required=False,
                               type=float, default=0.5, help='min_threshold_discard_htz_sample')
    compare_group.add_argument('--min_threshold_discard_htz_pos', required=False,
                               type=float, default=0.5, help='min_threshold_discard_htz_pos')
    compare_group.add_argument('--min_threshold_discard_all_pos', required=False,
                               type=float, default=0.5, help='min_threshold_discard_all_pos')
    compare_group.add_argument('--min_threshold_discard_all_sample', required=False,
                               type=float, default=0.5, help='min_threshold_discard_all_sample')
    compare_group.add_argument('-d', '--distance', default=15, required=False,
                               help='Minimun distance to cluster groups after comparison')

    arguments = parser.parse_args()

    return arguments
