#!/usr/bin/env python

# Standard library imports
import os
import sys
import re
import logging
import pandas as pd

# Third party imports
import argparse
import subprocess
import datetime


# Local application imports
from misc import extract_sample, check_create_dir, extract_read_list, file_to_list, obtain_group_cov_stats, clean_unwanted_files, check_reanalysis, remove_low_quality, obtain_overal_stats
from preprocessing import fastqc_quality
from bam_variants import run_snippy, extract_indels, merge_vcf, create_bamstat, create_coverage, run_snippy_core
from vcf_process import vcf_to_ivar_tsv, import_VCF4_core_to_compare
from annotation import annotate_snpeff, user_annotation, rename_reference_snpeff, report_samples_html, \
    user_annotation_aa, make_blast
from compare_snp_autosnippy import ddtb_compare, ddbb_create_intermediate, revised_df, recalibrate_ddbb_vcf_intermediate, \
    remove_position_range, extract_complex_list, identify_uncovered, extract_close_snps, remove_position_from_compare, remove_bed_positions, extract_only_snps, extract_bed_positions, extract_lowcov
from species_determination import mash_screen, kraken
from arguments import get_arguments

"""
=============================================================
HEADER
=============================================================

INSTITUTION:IiSGM
AUTHOR: Pedro J. Sola (pedroscampoy@gmail.com)
d^v^b
VERSION=0.1
CREATED: 22 Feb 2021
REVISION: 


TODO:
    Check file with multiple arguments
    Check program is installed (dependencies)
================================================================
END_OF_HEADER
================================================================
"""

# COLORS AND AND FORMATTING

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

logger = logging.getLogger()


def main():
    """
    Create main function to capture code errors: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    """

    args = get_arguments()

    ######################################################################
    #####################START PIPELINE###################################
    ######################################################################
    output = os.path.abspath(args.output)
    group_name = output.split("/")[-1]
    reference = os.path.abspath(args.reference)
    #annotation = os.path.abspath(args.annotation)

    # LOGGING
    # Create log file with date and time
    right_now = str(datetime.datetime.now())
    right_now_full = "_".join(right_now.split(" "))
    log_filename = group_name + "_" + right_now_full + ".log"
    log_folder = os.path.join(output, 'Logs')
    check_create_dir(log_folder)
    log_full_path = os.path.join(log_folder, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(message)s')

    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("\n\n" + BLUE + BOLD +
                "STARTING PIPELINE IN GROUP: " + group_name + END_FORMATTING)

    today = str(datetime.date.today())

    logger.info("ARGUMENTS:")
    logger.info(str(args))

    # Obtain all R1 and R2 from folder
    r1, r2 = extract_read_list(args.input_dir)

    # Check if there are samples to filter out
    sample_list_F = []
    if args.sample_list == None:
        logger.info("\n" + "No samples to filter")
        for r1_file, r2_file in zip(r1, r2):
            sample = extract_sample(r1_file, r2_file)
            sample_list_F.append(sample)
    else:
        logger.info("samples will be filtered")
        sample_list_F = file_to_list(args.sample_list)

    new_samples = check_reanalysis(args.output, sample_list_F)

    logger.info("\n%d samples will be analysed: %s" %
                (len(sample_list_F), ",".join(sample_list_F)))
    logger.info("\n%d NEW samples will be analysed: %s" %
                (len(new_samples), ",".join(new_samples)))
    #DECLARE FOLDERS CREATED IN PIPELINE ################
    #AND KEY FILES ######################################
    #####################################################
    # Annotation related parameters
    #script_dir = os.path.dirname(os.path.realpath(__file__))

    # Output related
    out_qc_dir = os.path.join(output, "Quality")
    out_qc_pre_dir = os.path.join(out_qc_dir, "raw")  # subfolder
    out_variant_dir = os.path.join(output, "Variants")
    out_core_dir = os.path.join(output, "Core")

    out_stats_dir = os.path.join(output, "Stats")
    out_stats_bamstats_dir = os.path.join(
        out_stats_dir, "Bamstats")  # subfolder
    out_stats_coverage_dir = os.path.join(
        out_stats_dir, "Coverage")  # subfolder
    out_compare_dir = os.path.join(output, "Compare")

    out_annot_dir = os.path.join(output, "Annotation")
    out_annot_snpeff_dir = os.path.join(out_annot_dir, "snpeff")  # subfolder
    out_annot_user_dir = os.path.join(out_annot_dir, "user")  # subfolder
    out_annot_user_aa_dir = os.path.join(out_annot_dir, "user_aa")  # subfolder
    out_annot_blast_dir = os.path.join(out_annot_dir, "blast")  # subfolder

    out_species_dir = os.path.join(output, "Species")
    new_sample_number = 0
    for r1_file, r2_file in zip(r1, r2):
        # EXtract sample name
        sample = extract_sample(r1_file, r2_file)
        args.sample = sample
        if sample in sample_list_F:
            # VARINAT SAMPLE DIR
            sample_variant_dir = os.path.join(out_variant_dir, sample)

            sample_number = str(sample_list_F.index(sample) + 1)
            sample_total = str(len(sample_list_F))
            if sample in new_samples:
                new_sample_number = str(int(new_sample_number) + 1)
                new_sample_total = str(len(new_samples))
                logger.info("\n" + WHITE_BG + "STARTING SAMPLE: " + sample +
                            " (" + sample_number + "/" + sample_total + ")" + " (" + new_sample_number + "/" + new_sample_total + ")" + END_FORMATTING)
            else:
                logger.info("\n" + WHITE_BG + "STARTING SAMPLE: " + sample +
                            " (" + sample_number + "/" + sample_total + ")" + END_FORMATTING)

            output_final_vcf = os.path.join(
                sample_variant_dir, 'snps.all.ivar.tsv')

            if not os.path.isfile(output_final_vcf):

                ##############START PIPELINE#####################
                #################################################

                # INPUT ARGUMENTS
                ################
                # check_file_exists(r1_file)
                # check_file_exists(r2_file)

                args.output = os.path.abspath(args.output)
                check_create_dir(args.output)

                # QUALITY CHECK in RAW with fastqc
                ######################################################
                check_create_dir(out_qc_dir)

                out_qc_raw_name_r1 = (".").join(r1_file.split(
                    '/')[-1].split('.')[0:-2]) + '_fastqc.html'
                out_qc_raw_name_r2 = (".").join(r2_file.split(
                    '/')[-1].split('.')[0:-2]) + '_fastqc.html'
                output_qc_raw_file_r1 = os.path.join(
                    out_qc_pre_dir, out_qc_raw_name_r1)
                output_qc_raw_file_r2 = os.path.join(
                    out_qc_pre_dir, out_qc_raw_name_r2)

                if os.path.isfile(output_qc_raw_file_r1) and os.path.isfile(output_qc_raw_file_r2):
                    logger.info(YELLOW + DIM + output_qc_raw_file_r1 +
                                " EXIST\nOmmiting QC for sample " + sample + END_FORMATTING)
                else:
                    logger.info(
                        GREEN + "Checking quality in sample " + sample + END_FORMATTING)
                    logger.info("R1: " + r1_file + "\nR2: " + r2_file)
                    fastqc_quality(r1_file, r2_file,
                                   out_qc_pre_dir, args.threads)

                """
                TODO: Human filter
                """

                # VARIANT CALLING WITH SNIPPY
                ###################################################

                output_vcf_sub = os.path.join(
                    sample_variant_dir, "snps.subs.vcf")
                output_vcf = os.path.join(sample_variant_dir, "snps.vcf")

                if os.path.isfile(output_vcf_sub) and os.path.isfile(output_vcf):
                    logger.info(YELLOW + DIM + output_vcf +
                                " EXIST\nOmmiting Variant calling in  " + sample + END_FORMATTING)
                else:
                    logger.info(
                        GREEN + "Calling variants with snippy " + sample + END_FORMATTING)

                    # There is a problem with the variant calling, freebayes in snippy is defined with a ploidy of 2, so it calls with 0/0, 0/1 and 1/1, filtering only the 1/1. This underestimates the call and can bring samples closer  phylogenetically when you really have more differences. The "envs/autosnippy/bin/snippy" file is modified:
                    # my $bcf_filter = qq{FMT/GT="1/1" && QUAL>=$minqual && FMT/DP>=$mincov && (FMT/AO)/(FMT/DP)>=$minfrac}; > my $bcf_filter = qq{FMT/GT="1/1" && QUAL>=$minqual && FMT/DP>=$mincov && (FMT/AO)/(FMT/DP)>=$minfrac | FMT/GT="0/1" && QUAL>=$minqual && FMT/DP>=$mincov && (FMT/AO)/(FMT/DP)>=$minfrac};

                    prior = datetime.datetime.now()
                    run_snippy(r1_file, r2_file, reference, out_variant_dir, sample,
                               threads=args.threads, minqual=10, minfrac=0.1, mincov=1)
                    old_bam = os.path.join(sample_variant_dir, "snps.bam")
                    old_bai = os.path.join(sample_variant_dir, "snps.bam.bai")
                    new_bam = os.path.join(sample_variant_dir, sample + ".bam")
                    new_bai = os.path.join(
                        sample_variant_dir, sample + ".bam.bai")
                    os.rename(old_bam, new_bam)
                    os.rename(old_bai, new_bai)
                    after = datetime.datetime.now()
                    print(("Done with function in: %s" % (after - prior)))

                #VARIANT FORMAT COMBINATION (REMOVE COMPLEX) ########
                #####################################################
                out_variant_indel_sample = os.path.join(
                    sample_variant_dir, "snps.indel.vcf")
                out_variant_all_sample = os.path.join(
                    sample_variant_dir, "snps.all.vcf")

                if os.path.isfile(out_variant_indel_sample):
                    logger.info(YELLOW + DIM + out_variant_indel_sample +
                                " EXIST\nOmmiting indel filtering in sample " + sample + END_FORMATTING)
                else:
                    logger.info(GREEN + "Filtering INDELS in " +
                                sample + END_FORMATTING)
                    extract_indels(output_vcf)

                if os.path.isfile(out_variant_all_sample):
                    logger.info(YELLOW + DIM + out_variant_all_sample +
                                " EXIST\nOmmiting vcf combination in sample " + sample + END_FORMATTING)
                else:
                    logger.info(GREEN + "Combining vcf in " +
                                sample + END_FORMATTING)
                    merge_vcf(output_vcf_sub, out_variant_indel_sample)

                #VARIANT FORMAT ADAPTATION TO IVAR ##################
                #####################################################
                out_variant_tsv_file = os.path.join(
                    sample_variant_dir, 'snps.all.ivar.tsv')

                if os.path.isfile(out_variant_tsv_file):
                    logger.info(YELLOW + DIM + out_variant_tsv_file +
                                " EXIST\nOmmiting format adaptation for sample " + sample + END_FORMATTING)
                else:
                    logger.info(
                        GREEN + "Adapting variants format in sample " + sample + END_FORMATTING)
                    vcf_to_ivar_tsv(out_variant_all_sample,
                                    out_variant_tsv_file)

            # SPECIES DETERMINATION
            ###################################################
            check_create_dir(out_species_dir)

            # Species determination with kraken2 and its standard database and visualization with ImportTaxonomy.pl from kronatools kit

            sample_species_dir = os.path.join(out_species_dir, sample)
            # print(sample_species_dir)
            krona_html = os.path.join(sample_species_dir + ".html")
            output_species = os.path.join(sample_species_dir + ".screen.tab")

            if args.kraken2_db != False:
                if os.path.isfile(krona_html):
                    logger.info(
                        YELLOW + krona_html + " EXIST\nOmmiting species determination with Kraken2 for " + sample + END_FORMATTING)
                else:
                    logger.info(
                        GREEN + "Species determination with Kraken2 for sample " + sample + END_FORMATTING)
                    kraken(r1_file, r2_file, sample_species_dir, args.kraken2_db,
                           krona_html, threads=args.threads)
            else:
                logger.info(
                    YELLOW + BOLD + "No Kraken database suplied, skipping specie assignation in group " + group_name + END_FORMATTING)

            # Species determination with mash and its bacterial database

            if args.mash_db != False:
                if os.path.isfile(output_species):
                    logger.info(
                        YELLOW + output_species + " EXIST\nOmmiting species determination with Mash screen for " + sample + END_FORMATTING)
                else:
                    logger.info(
                        GREEN + "Species determination with Mash for sample " + sample + END_FORMATTING)

                    mash_screen(r1_file, out_species_dir, r2_file=r2_file, winner=True,
                                threads=args.threads, mash_database=args.mash_db)

                    # Name the columns of the mash output and sort them in descending order by identity
                    output_sort_species = pd.read_csv(output_species, sep='\t', header=None, names=[
                                                      'Identity', 'Share-hashes', 'Median-multiplicity', 'p-value', 'ID accession', 'Organism']).sort_values(by=['Identity'], ascending=False)
                    output_sort_species.to_csv(
                        output_species, sep='\t', index=None)
            else:
                logger.info(
                    YELLOW + BOLD + "No Mash database suplied, skipping specie assignation in group " + group_name + END_FORMATTING)

            ########################CREATE STATS AND QUALITY FILTERS########################################################################
            ################################################################################################################################
            #CREATE Bamstats#######################################
            #######################################################
            check_create_dir(out_stats_dir)
            check_create_dir(out_stats_bamstats_dir)
            out_bamstats_name = sample + ".bamstats"
            out_bamstats_file = os.path.join(
                out_stats_bamstats_dir, out_bamstats_name)
            bam_sample_file = os.path.join(sample_variant_dir, sample + ".bam")

            if os.path.isfile(out_bamstats_file):
                logger.info(YELLOW + DIM + out_bamstats_file +
                            " EXIST\nOmmiting Bamstats for  sample " + sample + END_FORMATTING)
            else:
                logger.info(GREEN + "Creating bamstats in sample " +
                            sample + END_FORMATTING)
                create_bamstat(
                    bam_sample_file, out_stats_bamstats_dir, sample, threads=args.threads)

            #CREATE Bamstats#######################################
            #######################################################
            check_create_dir(out_stats_coverage_dir)
            out_coverage_name = sample + ".cov"
            out_coverage_file = os.path.join(
                out_stats_coverage_dir, out_coverage_name)

            if os.path.isfile(out_coverage_file):
                logger.info(YELLOW + DIM + out_coverage_file +
                            " EXIST\nOmmiting Bamstats for  sample " + sample + END_FORMATTING)
            else:
                logger.info(GREEN + "Creating coverage in sample " +
                            sample + END_FORMATTING)
                create_coverage(bam_sample_file,
                                out_stats_coverage_dir, sample)

    # coverage OUTPUT SUMMARY
    ######################################################
    prior_recal = datetime.datetime.now()
    logger.info(GREEN + "Creating summary report for coverage result in group " +
                group_name + END_FORMATTING)
    obtain_group_cov_stats(out_stats_dir, group_name)
    after_recal = datetime.datetime.now()
    logger.info("Done with report for coverage: %s" %
                (after_recal - prior_recal))

    # READS and VARIANTS OUTPUT SUMMARY
    ######################################################
    logger.info(GREEN + "Creating overal summary report in group " +
                group_name + END_FORMATTING)
    obtain_overal_stats(output, group_name)

    # REMOVE UNCOVERED
    ##############################################################################################################################
    logger.info(GREEN + "Removing low quality samples in group " +
                group_name + END_FORMATTING)
    uncovered_samples = remove_low_quality(
        output, cov20=args.coverage20, unmapped_per=args.unmapped, min_hq_snp=args.min_snp, type_remove='Uncovered')

    if len(uncovered_samples) > 1:
        logger.info(GREEN + "Uncovered samples: " +
                    (",").join(uncovered_samples) + END_FORMATTING)
    else:
        logger.info(GREEN + "NO uncovered samples found" + END_FORMATTING)

    # RUN SNIPPY CORE
    ##############################################################################################################################
    if args.core:
        check_create_dir(out_core_dir)
        logger.info(GREEN + "Running snippy-core " +
                    group_name + END_FORMATTING)
        run_snippy_core(out_variant_dir, out_core_dir, reference)

        logger.info(GREEN + "Adapting core-snp to compare format " +
                    group_name + END_FORMATTING)
        core_vcf_file = os.path.join(out_core_dir, "core.vcf")
        core_vcf_file_adapted = os.path.join(
            out_core_dir, "core.vcf.adapted.tsv")
        core_vcf_file_removed = os.path.join(
            out_core_dir, "core.vcf.adapted.final.tsv")

        core_vcf_df_adapted = import_VCF4_core_to_compare(core_vcf_file)
        core_vcf_df_adapted.to_csv(
            core_vcf_file_adapted, sep="\t", index=False)

        logger.info(GREEN + "Obtaining clustered positions " +
                    group_name + END_FORMATTING)

        close_positions_list = extract_close_snps(
            core_vcf_df_adapted, snps_in_10=1)
        logger.info(GREEN + "Obtaining uncovered positions " +
                    group_name + END_FORMATTING)
        uncovered_list = identify_uncovered(
            out_stats_coverage_dir, min_coverage=10, nocall_fr=0.5)

        logger.debug('Clustered positions in core SNP:\n{}'.format(
            (",".join([str(x) for x in close_positions_list]))))
        logger.debug('Uncovered positions in all samples:\n{}'.format(
            (",".join([str(x) for x in uncovered_list]))))

        to_remove_list = close_positions_list + uncovered_list

        remove_df = remove_position_from_compare(
            core_vcf_df_adapted, to_remove_list)
        remove_df.to_csv(core_vcf_file_removed, sep="\t", index=False)

        ddtb_compare(core_vcf_file_removed, distance=args.distance)

    #ANNOTATION WITH SNPEFF AND USER INPUT ##############
    #####################################################
    logger.info("\n\n" + BLUE + BOLD + "STARTING ANNOTATION IN GROUP: " +
                group_name + END_FORMATTING + "\n")
    check_create_dir(out_annot_dir)
    check_create_dir(out_annot_snpeff_dir)
    # SNPEFF
    if args.snpeff_database != False:
        for root, _, files in os.walk(out_variant_dir):
            for name in files:
                if name == 'snps.all.vcf':
                    sample = root.split('/')[-1]
                    filename = os.path.join(root, name)
                    chrom_filename = os.path.join(
                        root, 'snps.all.chromosome.vcf')
                    out_annot_file = os.path.join(
                        out_annot_snpeff_dir, sample + ".annot")
                    if os.path.isfile(out_annot_file):
                        logger.info(YELLOW + DIM + out_annot_file +
                                    " EXIST\nOmmiting snpEff Annotation for sample " + sample + END_FORMATTING)
                    else:
                        logger.info(
                            GREEN + "Annotating sample with snpEff: " + sample + END_FORMATTING)
                        rename_reference_snpeff(filename, chrom_filename)
                        annotate_snpeff(chrom_filename, out_annot_file,
                                        database=args.snpeff_database)
    else:
        logger.info(YELLOW + DIM + " No SnpEff database suplied, skipping annotation in group " +
                    group_name + END_FORMATTING)
    # USER DEFINED
    if not args.annot_bed and not args.annot_vcf:
        logger.info(
            YELLOW + BOLD + "Ommiting User Annotation, no BED or VCF files supplied" + END_FORMATTING)
    else:
        check_create_dir(out_annot_user_dir)
        for root, _, files in os.walk(out_variant_dir):
            for name in files:
                if name == 'snps.all.ivar.tsv':
                    sample = root.split('/')[-1]
                    logger.info(
                        'User bed/vcf annotation in sample {}'.format(sample))
                    filename = os.path.join(root, name)
                    out_annot_file = os.path.join(
                        out_annot_user_dir, sample + ".tsv")
                    user_annotation(
                        filename, out_annot_file, vcf_files=args.annot_vcf, bed_files=args.annot_bed)

    # USER AA DEFINED
    if not args.annot_aa:
        logger.info(
            YELLOW + BOLD + "Ommiting User aa Annotation, no AA files supplied" + END_FORMATTING)
    else:
        check_create_dir(out_annot_user_aa_dir)
        for root, _, files in os.walk(out_annot_snpeff_dir):
            if root == out_annot_snpeff_dir:
                for name in files:
                    if name.endswith('.annot'):
                        sample = name.split('.')[0]
                        logger.info(
                            'User aa annotation in sample {}'.format(sample))
                        filename = os.path.join(root, name)
                        out_annot_aa_file = os.path.join(
                            out_annot_user_aa_dir, sample + ".tsv")
                        if os.path.isfile(out_annot_aa_file):
                            user_annotation_aa(
                                out_annot_aa_file, out_annot_aa_file, aa_files=args.annot_aa)
                        else:
                            user_annotation_aa(
                                filename, out_annot_aa_file, aa_files=args.annot_aa)
    # USER FASTA ANNOTATION
    if not args.annot_fasta:
        logger.info(
            YELLOW + BOLD + "Ommiting User FASTA Annotation, no FASTA files supplied" + END_FORMATTING)
    else:
        check_create_dir(out_annot_blast_dir)
        for root, _, files in os.walk(out_variant_dir):
            for name in files:
                if name.endswith('.consensus.subs.fa'):
                    filename = os.path.join(root, name)
                    sample = root.split('/')[-1]
                    logger.info(
                        'User FASTA annotation in sample {}'.format(sample))
                    # out_annot_aa_file = os.path.join(
                    #    out_annot_user_aa_dir, sample + ".tsv")
                    for db in args.annot_fasta:
                        make_blast(filename, db, sample, out_annot_blast_dir,
                                   db_type="nucl", query_type="nucl", evalue=0.0001, threads=8)

    # USER AA TO HTML
    # if not args.annot_aa:
    #     logger.info(
    #         YELLOW + BOLD + "Ommiting User aa Annotation to HTML, no AA files supplied" + END_FORMATTING)
    # else:
    #     annotated_samples = []
    #     logger.info('Adapting annotation to html in {}'.format(group_name))
    #     for root, _, files in os.walk(out_annot_user_aa_dir):
    #         if root == out_annot_user_aa_dir:
    #             for name in files:
    #                 if name.endswith('.tsv'):
    #                     sample = name.split('.')[0]
    #                     annotated_samples.append(sample)
    #                     filename = os.path.join(root, name)
    #                     annotation_to_html(filename, sample)
    #     annotated_samples = [str(x) for x in annotated_samples]
    #     report_samples_html_all = report_samples_html.replace(
    #         'ALLSAMPLES', ('","').join(annotated_samples))  # NEW
    #     with open(os.path.join(out_annot_user_aa_dir, '00_all_samples.html'), 'w+') as f:
    #         f.write(report_samples_html_all)

    # SNP COMPARISON using tsv variant files
    ######################################################
    logger.info("\n\n" + BLUE + BOLD + "STARTING COMPARISON IN GROUP: " +
                group_name + END_FORMATTING + "\n")

    check_create_dir(out_compare_dir)
    folder_compare = today + "_" + group_name
    path_compare = os.path.join(out_compare_dir, folder_compare)
    check_create_dir(path_compare)
    full_path_compare = os.path.join(path_compare, group_name)

    compare_snp_matrix_recal = full_path_compare + ".revised.final.tsv"
    compare_snp_matrix_recal_intermediate = full_path_compare + ".revised_intermediate.tsv"
    compare_snp_matrix_recal_mpileup = full_path_compare + \
        ".revised_intermediate_vcf.tsv"
    compare_snp_matrix_INDEL_intermediate = full_path_compare + \
        ".revised_INDEL_intermediate.tsv"
    compare_only_snps = full_path_compare + "_ONLY_SNPs.revised.tsv"

    # Create intermediate

    recalibrated_snp_matrix_intermediate = ddbb_create_intermediate(
        out_variant_dir, out_stats_coverage_dir, min_freq_discard=0.1, min_alt_dp=8, only_snp=False)
    # recalibrated_snp_matrix_intermediate.to_csv(
    #     compare_snp_matrix_recal_intermediate, sep="\t", index=False)

    # Remove SNPs from BED file (PE/PPE)

    if args.remove_bed:
        recalibrated_snp_matrix_intermediate = remove_bed_positions(
            recalibrated_snp_matrix_intermediate, args.remove_bed, full_path_compare)

    recalibrated_snp_matrix_intermediate.to_csv(
        compare_snp_matrix_recal_intermediate, sep="\t", index=False)

    # Recalibrate intermediate with VCF

    prior_recal = datetime.datetime.now()
    recalibrated_snp_matrix_mpileup = recalibrate_ddbb_vcf_intermediate(
        compare_snp_matrix_recal_intermediate, out_variant_dir, min_cov_low_freq=12)
    recalibrated_snp_matrix_mpileup.to_csv(
        compare_snp_matrix_recal_mpileup, sep="\t", index=False)

    after_recal = datetime.datetime.now()
    logger.debug("Done with recalibration vcf: %s" %
                 (after_recal - prior_recal))

    # Remove SNPs located within INDELs

    compare_snp_matrix_INDEL_intermediate_df = remove_position_range(
        recalibrated_snp_matrix_mpileup)
    compare_snp_matrix_INDEL_intermediate_df.to_csv(
        compare_snp_matrix_INDEL_intermediate, sep="\t", index=False)

    # Extract all positions marked as complex
    complex_variants = extract_complex_list(out_variant_dir)
    logger.debug('Complex positions in all samples:\n{}'.format(
        (",".join([str(x) for x in complex_variants]))))

    # Extract all low coverage o not covered positions

    prior = datetime.datetime.now()

    symbol_file = full_path_compare + '_symbol_lowcov.tsv'
    symbol_lowcov = extract_lowcov(compare_snp_matrix_INDEL_intermediate) # It is made by the INDEL_intermediate.tsv, taking 0 and 1 into account, can also be made with intermediate.highfreq.tsv
    symbol_lowcov.to_csv(symbol_file, sep='\t', index=False)

    after = datetime.datetime.now()
    print(("Done with function extract_lowcov in: %s" %
               (after - prior) + "\n"))

    # Clean all faulty positions and samples => Final table

    recalibrated_revised_INDEL_df = revised_df(compare_snp_matrix_INDEL_intermediate_df,
                                               path_compare,
                                               complex_pos=complex_variants,
                                               min_freq_include=0.8,
                                               min_threshold_discard_uncov_sample=args.min_threshold_discard_uncov_sample,
                                               min_threshold_discard_uncov_pos=args.min_threshold_discard_uncov_pos,
                                               min_threshold_discard_htz_sample=args.min_threshold_discard_htz_sample,
                                               min_threshold_discard_htz_pos=args.min_threshold_discard_htz_pos,
                                               min_threshold_discard_all_pos=args.min_threshold_discard_all_pos,
                                               min_threshold_discard_all_sample=args.min_threshold_discard_all_sample,
                                               remove_faulty=True,
                                               drop_samples=True,
                                               drop_positions=True,
                                               windows_size_discard=args.window)

    recalibrated_revised_df = recalibrated_revised_INDEL_df[~recalibrated_revised_INDEL_df['Position'].isin(symbol_lowcov['Position'])]

    recalibrated_revised_df.to_csv(
            compare_snp_matrix_recal, sep='\t', index=False)

    if args.only_snp:
        compare_only_snps_df = extract_only_snps(
            compare_snp_matrix_recal)
        compare_only_snps_df.to_csv(compare_only_snps, sep="\t", index=False)

    # Matrix to pairwise and mwk

    ddtb_compare(compare_snp_matrix_recal, distance=args.distance)

    if args.only_snp:
        ddtb_compare(compare_only_snps, distance=args.distance)

    # Annotated SNPs from BED file (genes or positions of interest)

    if args.extract_bed:
        annotated_snps_final = extract_bed_positions(
            recalibrated_revised_INDEL_df, args.extract_bed, full_path_compare)

    logger.info("\n\n" + MAGENTA + BOLD + "COMPARING FINISHED IN GROUP: " +
                group_name + END_FORMATTING + "\n")

    logger.info("\n\n" + MAGENTA + BOLD +
                "#####END OF PIPELINE AUTOSNIPPY ANALYSIS#####" + END_FORMATTING + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise
