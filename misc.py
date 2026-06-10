
import os
import sys
import re
import subprocess
import shutil
import logging
import datetime
import pandas as pd
import numpy as np
from statistics import mean
from pandarallel import pandarallel
import concurrent.futures


logger = logging.getLogger()

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


def check_file_exists(file_name):

    """
    Check file exist and is not 0 Kb, if not program exit.
    """

    file_info = os.stat(file_name)

    if not os.path.isfile(file_name) or file_info.st_size == 0:
        logger.info(RED + BOLD + "File: %s not found or empty\n" % file_name + END_FORMATTING)
        sys.exit(1)

    return os.path.isfile(file_name)


def extract_sample(R1_file, R2_file):

    """
    Extract sample from R1, R2 files.
    """

    basename_R1 = os.path.basename(R1_file)
    basename_R2 = os.path.basename(R2_file)

    sample_name_R = os.path.commonprefix([basename_R1, basename_R2])

    long_suffix = re.search('_S.*', sample_name_R)
    short_suffix = re.search('_R.*', sample_name_R)
    bar_suffix = re.search('_$', sample_name_R)
    dot_suffix = re.search('.R$', sample_name_R)

    if long_suffix:
        match = long_suffix.group()
        sample_name = sample_name_R.split(match)[0]
    elif short_suffix:
        match = short_suffix.group()
        sample_name = sample_name_R.split(match)[0]
    elif bar_suffix:
        match = bar_suffix.group()
        sample_name = sample_name_R.rstrip("_")
    elif dot_suffix:
        match = dot_suffix.group()
        sample_name = sample_name_R.rstrip(".R")
    else:
        sample_name = sample_name_R

    return sample_name


def check_create_dir(path):

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def execute_subprocess(cmd, isShell=False):

    """
    https://crashcourse.housegordon.org/python-subprocess.html
    https://docs.python.org/3/library/subprocess.html 
    Execute and handle errors with subprocess, outputting stderr instead of the subprocess CalledProcessError
    """

    logger.debug("")
    logger.debug(cmd)

    if cmd[0] == "java":
        prog = cmd[2].split("/")[-1] + " " + cmd[3]
        param = cmd[4:]
    elif cmd[0] == "samtools" or cmd[0] == "bwa" or cmd[0] == "gatk":
        prog = " ".join(cmd[0:2])
        param = cmd[3:]
    else:
        prog = cmd[0]
        param = cmd[1:]

    try:
        command = subprocess.run(
            cmd, shell=isShell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if command.returncode == 0:
            logger.debug(GREEN + DIM + "Program %s successfully executed" % prog + END_FORMATTING)
        else:
            logger.info(RED + BOLD + "Command %s FAILED\n" % prog + END_FORMATTING
                        + BOLD + "WITH PARAMETERS: " +
                        END_FORMATTING + " ".join(param) + "\n"
                        + BOLD + "EXIT-CODE: %d\n" % command.returncode +
                        "ERROR:\n" + END_FORMATTING + command.stderr.decode().strip())
        logger.debug(command.stdout)
        logger.debug(command.stderr.decode().strip())
    except OSError as e:
        sys.exit(RED + BOLD + "failed to execute program '%s': %s" % (prog, str(e)) + END_FORMATTING)


def extract_read_list(input_dir):

    """
    Search files in a directory sort by name and extract comon name of R1 and R2
    with extract_sample() function
    190615 - Limit only parent folder, not subdirectories
    """

    input_dir = os.path.abspath(input_dir)
    all_fasta = []
    r1_list = []
    r2_list = []

    for root, _, files in os.walk(input_dir):
        if root == input_dir:  # This only apply to parent folder, not subdirectories
            for name in files:
                filename = os.path.join(root, name)
                is_fasta = re.match(r'.*\.f(ast)*[aq](\.gz)*', filename)
                if is_fasta:
                    all_fasta.append(filename)

    all_fasta = sorted(all_fasta)

    if len(all_fasta) % 2 == 0:
        for index, fasta_file in enumerate(all_fasta):
            if index % 2 == 0:
                r1_list.append(fasta_file)
            elif index % 1 == 0:
                r2_list.append(fasta_file)
    else:
        logger.info('ERROR: The number of fastq sequence are not paired')

    r1_list = sorted(r1_list)
    r2_list = sorted(r2_list)

    return r1_list, r2_list


def file_to_list(file_name):

    list_F = []
    file_name_abs = os.path.abspath(file_name)

    with open(file_name_abs, "r") as f:
        for line in f:
            list_F.append(line.strip())
    return list_F


def calculate_cov_stats(file_cov):

    sample = file_cov.split("/")[-1].split(".")[0]
    df = pd.read_csv(file_cov, sep="\t", names=["#CHROM", "POS", "COV"])

    unmapped_pos = len(df.POS[df.COV == 0].tolist())
    pos_0_10 = len(df.POS[(df.COV > 0) & (df.COV <= 10)].tolist())
    pos_high10 = len(df.POS[(df.COV > 10)].tolist())
    pos_high20 = len(df.POS[(df.COV > 20)].tolist())
    pos_high30 = len(df.POS[(df.COV > 30)].tolist())
    pos_high50 = len(df.POS[(df.COV > 50)].tolist())
    pos_high100 = len(df.POS[(df.COV >= 100)].tolist())

    total_pos = df.shape[0]

    unmapped_prop = "%.2f" % ((unmapped_pos/total_pos)*100)
    prop_0_10 = "%.2f" % ((pos_0_10/total_pos)*100)
    prop_high10 = "%.2f" % ((pos_high10/total_pos)*100)
    prop_high20 = "%.2f" % ((pos_high20/total_pos)*100)
    prop_high30 = "%.2f" % ((pos_high30/total_pos)*100)
    prop_high50 = "%.2f" % ((pos_high50/total_pos)*100)
    prop_high100 = "%.2f" % ((pos_high100/total_pos)*100)

    mean_cov = "%.2f" % (df.COV.mean())

    return sample, mean_cov, unmapped_prop, prop_0_10, prop_high10, prop_high20, prop_high30, prop_high50, prop_high100


def obtain_group_cov_stats(directory, group_name):

    directory_path = os.path.abspath(directory)
    samples_to_skip = []
    previous_stat = False

    output_group_name = group_name + ".coverage.summary.tab"
    output_file = os.path.join(directory_path, output_group_name)

    if os.path.exists(output_file):
        previous_stat = True
        df_stat = pd.read_csv(output_file, sep="\t")
        samples_to_skip = df_stat["#SAMPLE"].tolist()
        logger.debug("Skipped samples for coverage calculation:" +
                     (",").join(str(samples_to_skip)))

    columns = ["#SAMPLE", "MEAN_COV", "UNMAPPED_PROP", "COV1-10X", "COV>10X", "COV>20X", "COV>30X", "COV>50X", "COV>100X"]

    files_list = []

    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.cov'):
                filename = os.path.join(root, name)
                sample = name.split(".")[0]
                if not sample in samples_to_skip:
                    files_list.append(filename)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        dfs = executor.map(calculate_cov_stats, files_list)
    df = pd.DataFrame(dfs, columns=columns)

    if previous_stat:
        df = pd.concat([df_stat, df], ignore_index=True, sort=True)
        df.to_csv(output_file, sep="\t", index=False)
    else:
        df.to_csv(output_file, sep="\t", index=False)


def extract_snp_count(output_dir, sample):

    sample = str(sample)

    if '.' in sample:
        sample = sample.split('.')[0]
    variants_folder = os.path.join(output_dir, 'Variants')
    raw_var_folder = os.path.join(variants_folder, sample)
    filename = os.path.join(raw_var_folder, "snps.all.ivar.tsv")

    if os.path.exists(filename):
        df = pd.read_csv(filename, sep="\t")
        df = df.drop_duplicates(subset=['POS', 'REF', 'ALT'], keep="first")
        high_quality_snps = df["POS"][(df.TOTAL_DP >= 20) &
                                      (df.ALT_FREQ >= 0.8) &
                                      (df.TYPE == 'snp') &
                                      ~(df.OLDVAR.isin(['complex', 'mnp']))].tolist()

        htz_snps = df["POS"][(df.TOTAL_DP >= 20) &
                             (df.ALT_FREQ < 0.8) &
                             (df.ALT_FREQ >= 0.2) &
                             (df.TYPE == 'snp') &
                             ~(df.OLDVAR.isin(['complex', 'mnp']))].tolist()

        indels = df["POS"][(df.TOTAL_DP >= 20) &
                           (df.ALT_FREQ >= 0.8) &
                           ((df.TYPE == 'ins') | (df.TYPE == 'del'))].tolist()

        return (len(high_quality_snps), len(htz_snps), len(indels))
    else:
        logger.debug("FILE " + filename + " NOT FOUND")
        return None


def extract_mapped_reads(output_dir, sample):

    sample = str(sample)

    if '.' in sample:
        sample = sample.split('.')[0]
    stats_folder = os.path.join(output_dir, 'Stats')
    bamstats_folder = os.path.join(stats_folder, 'Bamstats')
    filename = os.path.join(bamstats_folder, sample + ".bamstats")

    if os.path.exists(filename):
        reads_mapped = 0
        mappep_percentage = 0
        properly_paired = 0
        paired_percentage = 0
        with open(filename, 'r') as f:
            logger.debug('File bamstat: {}'.format(filename))
            for line in f:
                if 'mapped' in line and '%' in line:
                    reads_mapped = line.split(" ")[0]
                    try:
                        mappep_percentage = line.split("(")[-1].split("%")[0]
                        mappep_percentage = float(mappep_percentage)
                    except ValueError:  # To avoid (NA: NA)
                        mappep_percentage = 0

                elif 'properly paired' in line:
                    properly_paired = line.split(" ")[0]
                    try:
                        paired_percentage = line.split("(")[-1].split("%")[0]
                        paired_percentage = float(paired_percentage)
                    except ValueError:  # To avoid (NA: NA)
                        paired_percentage = 0

        if len([x for x in [reads_mapped, mappep_percentage, properly_paired, paired_percentage] if x != 0]):
            return int(reads_mapped), float(mappep_percentage), int(properly_paired), float(paired_percentage)
        else:
            return 0, 0, 0, 0
    else:
        logger.info("FILE " + filename + " NOT FOUND")
        return None


# def extract_n_consensus(output_dir, sample):
#     sample = str(sample)

#     if '.' in sample:
#         sample = sample.split('.')[0]
#     consensus_folder = os.path.join(output_dir, 'Variants')
#     consensus_folder_sample = os.path.join(consensus_folder, sample)
#     filename = os.path.join(consensus_folder_sample, "snps.consensus.fa")

#     if os.path.exists(filename):
#         with open(filename, 'r') as f:
#             content = f.read()
#             content_list = content.split('\n')
#             #sample_fq = content_list[0].strip(">")
#             # In case fasta is in several lines(not by default)
#             sequence = ("").join(content_list[1:]).strip()
#             all_N = re.findall(r'N+', sequence)
#             if all_N:
#                 leading_N = re.findall(r'^N+', sequence)
#                 tailing_N = re.findall(r'N+$', sequence)
#                 length_N = [len(x) for x in all_N]
#                 individual_N = [x for x in length_N if x == 1]
#                 mean_length_N = mean(length_N)
#                 sum_length_N = sum(length_N)
#                 total_perc_N = sum_length_N / len(sequence) * 100
#                 return(len(all_N), len(individual_N), len(leading_N), len(tailing_N), sum_length_N, total_perc_N, mean_length_N)
#             else:
#                 return(0, 0, 0, 0, 0, 0, 0)

#     else:
#         logger.info("FILE " + filename + " NOT FOUND")
#         return None


def obtain_overal_stats(output_dir, group):

    pandarallel.initialize()
    samples_to_skip = []
    previous_stat = False

    stat_folder = os.path.join(output_dir, 'Stats')
    overal_stat_file = os.path.join(stat_folder, group + ".overal.stats.tab")

    columns = ['#SAMPLE', 'MEAN_COV', 'UNMAPPED_PROP', 'COV1-10X', 'COV>10X', 'COV>20X', 'COV>30X', 'COV>50X', 'COV>100X']

    if os.path.exists(overal_stat_file):
        previous_stat = True
        df_stat = pd.read_csv(overal_stat_file, sep="\t")
        samples_to_skip = df_stat["#SAMPLE"].tolist()
        logger.debug("Skipped samples for coverage calculation:" + (",").join(str(samples_to_skip)))

    for root, _, files in os.walk(stat_folder):
        for name in files:
            if name.endswith('coverage.summary.tab'):
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep="\t")
                df = df[~df["#SAMPLE"].isin(samples_to_skip)]
                if df.shape[0] > 0:
                    df[['HQ_SNP', 'HTZ_SNP', 'INDELS']] = df.parallel_apply(lambda x: extract_snp_count(output_dir, x['#SAMPLE']), axis=1, result_type="expand")
                    df[['mapped_reads', 'perc_mapped', 'paired_mapped', 'perc_paired']] = df.parallel_apply(lambda x: extract_mapped_reads(output_dir, x['#SAMPLE']), axis=1, result_type="expand")

    if previous_stat:
        df = pd.concat([df_stat, df], ignore_index=True, sort=True)
        df = df[columns + [col for col in df.columns if col != "#SAMPLE" and col != "MEAN_COV" and col != "UNMAPPED_PROP" and col != "COV1-10X" and col != "COV>10X" and col != "COV>20X" and col != "COV>30X" and col != "COV>50X" and col != "COV>100X"]]
        df.to_csv(overal_stat_file, sep="\t", index=False)

    else:
        df = df[columns + [col for col in df.columns if col != "#SAMPLE" and col != "MEAN_COV" and col != "UNMAPPED_PROP" and col != "COV1-10X" and col != "COV>10X" and col != "COV>20X" and col != "COV>30X" and col != "COV>50X" and col != "COV>100X"]]
        df.to_csv(overal_stat_file, sep="\t", index=False)


def remove_low_quality(output_dir, cov20=70, unmapped_per=25, min_hq_snp=8, type_remove='Uncovered'):

    right_now = str(datetime.datetime.now())
    right_now_full = "_".join(right_now.split(" "))
    output_dir = os.path.abspath(output_dir)
    uncovered_dir = os.path.join(output_dir, type_remove)  # Uncovered or Mixed
    uncovered_dir_variants = os.path.join(uncovered_dir, "Variants")

    check_create_dir(uncovered_dir)
    uncovered_samples = []

    for root, _, files in os.walk(output_dir):
        if root.endswith('Stats'):
            for name in files:
                filename = os.path.join(root, name)
                if name.endswith('overal.stats.tab'):
                    coverage_stat_file = filename
                    stats_df = pd.read_csv(coverage_stat_file, sep="\t")
                    stats_df = stats_df.fillna(0)

                    stats_df['HQ_SNP'] = stats_df['HQ_SNP'].astype(str)

                    def f(x): return x if x.replace('.', '', 1).isdigit() else max(x.strip("()").split(","))

                    stats_df['HQ_SNP'] = stats_df.apply(lambda x: f(x.HQ_SNP), axis=1)

                    stats_df['HQ_SNP'] = stats_df['HQ_SNP'].astype(float)
                    uncovered_samples = stats_df["#SAMPLE"][(stats_df['COV>20X'] <= cov20) | (stats_df["UNMAPPED_PROP"] >= unmapped_per) | (stats_df["HQ_SNP"] < min_hq_snp)].tolist()

                    # create a df with only covered to replace the original
                    covered_df = stats_df[~stats_df['#SAMPLE'].isin(uncovered_samples)]
                    covered_df.to_csv(coverage_stat_file, sep="\t", index=False)

                    # create a df with uncovered
                    uncovered_df = stats_df[stats_df['#SAMPLE'].isin(uncovered_samples)]
                    uncovered_table_filename = right_now_full + '_uncovered.summary.tab'
                    uncovered_table_file = os.path.join(uncovered_dir, uncovered_table_filename)

                    if len(uncovered_samples) > 0:
                        uncovered_df.to_csv(uncovered_table_file, sep="\t", index=False)
                elif name.endswith('.coverage.summary.tab'):
                    covstats_df = pd.read_csv(filename, sep="\t")
                    final_covstat = filename

    uncovered_samples = [str(x) for x in uncovered_samples]
    def_covstats_df = covstats_df[~covstats_df['#SAMPLE'].isin(uncovered_samples)]
    def_covstats_df.to_csv(final_covstat, sep="\t", index=False)

    logger.debug("Uncovered_samples:")
    logger.debug(uncovered_samples)

    # Remove other files
    for root, _, files in os.walk(output_dir):
        for name in files:
            if name.endswith('.cov') or name.endswith('.bamstats'):
                filename = os.path.join(root, name)
                sample = name.split(".")[0]
                if sample in uncovered_samples:
                    logger.debug(
                        "Removing FAULTY file {}".format(filename))
                    os.remove(filename)

    sample_list_F = []

    r1, r2 = extract_read_list(output_dir)
    for r1_file, r2_file in zip(r1, r2):
        sample = extract_sample(r1_file, r2_file)
        sample_list_F.append(sample)

    # MOVE Fastq
    if len(uncovered_samples) > 0:
        for uncovered_sample in uncovered_samples:
            try:
                uncovered_index = sample_list_F.index(uncovered_sample)
                destination_file_r1 = os.path.join(uncovered_dir, r1[uncovered_index].split("/")[-1])
                destination_file_r2 = os.path.join(uncovered_dir, r2[uncovered_index].split("/")[-1])
                logger.debug("Moving FAULTY fastas {} AND {} TO {} AND {}".format(r1[uncovered_index], r2[uncovered_index], destination_file_r1, destination_file_r2))
                shutil.move(r1[uncovered_index], destination_file_r1)
                shutil.move(r2[uncovered_index], destination_file_r2)
            except:
                logger.info('ERROR: No uncovered detected')

    # Move Variant Folder
    for root, _, files in os.walk(output_dir):
        if root.endswith('Variants') and not "Uncovered" in root:
            if len(uncovered_samples) > 0:
                for uncovered in uncovered_samples:
                    filename = os.path.join(root, uncovered)
                    destination_file = os.path.join(uncovered_dir_variants, uncovered)
                    if not os.path.exists(destination_file):
                        logger.debug("Moving FAULTY folder {} TO {}".format(filename, destination_file))
                        shutil.move(filename, destination_file)
                    else:
                        logger.debug("{} already exist".format(destination_file))

    return uncovered_samples


def check_reanalysis(output_dir, samples_to_analyze):

    output_dir = os.path.abspath(output_dir)
    new_samples = []

    variant_dir = os.path.join(output_dir, "Variants")
    compare_dir = os.path.join(output_dir, "Compare")

    previous_files = [variant_dir, compare_dir]

    # check how many folders exist
    file_exist = sum([os.path.exists(x) for x in previous_files])  # True = 1, False = 0

    # Handle reanalysis: First time; reanalysis o reanalysis with aditional samples
    if file_exist > 0: 
        previous_samples_list = os.listdir(variant_dir)

        if len(samples_to_analyze) == len(previous_samples_list):
            logger.info(MAGENTA + "\nPREVIOUS ANALYSIS DETECTED, NO NEW SEQUENCES ADDED\n" + END_FORMATTING)
        else:
            new_samples = set(samples_to_analyze) - set(previous_samples_list)
            logger.info(MAGENTA + "\nPREVIOUS ANALYSIS DETECTED, " + str(len(new_samples)) + " NEW SEQUENCES ADDED\n" + END_FORMATTING)

    return list(new_samples)


# ── Preprocessing ────────────────────────────────────────────────────────

def fastqc_quality(r1, r2, output_dir, threads=8):

    check_create_dir(output_dir)

    cmd = ['fastqc', r1, r2, '-o', output_dir,'--threads', str(threads)]
    execute_subprocess(cmd)


# ── VCF process ──────────────────────────────────────────────────────────

def import_VCF4_core_to_compare(vcf_file, sep='\t'):

    header_lines = 0

    with open(vcf_file) as f:
        first_line = f.readline().strip()
        next_line = f.readline().strip()
        while next_line.startswith("##"):
            header_lines = header_lines + 1
            next_line = f.readline()

    if first_line.endswith('VCFv4.2'):

        # Use first line as header
        df = pd.read_csv(vcf_file, sep=sep, skiprows=[
            header_lines], header=header_lines)

        df.POS = df.POS.apply(str)
        df['Position'] = df["#CHROM"] + "|" + \
            df["REF"] + "|" + df["POS"] + "|" + df["ALT"]
        df = df.drop(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis=1)
        df = df[['Position'] + [col for col in df.columns if col not in ['Position']]]
        df["N"] = df.apply(lambda x: sum([i != 0 for i in x[1:]]), axis=1)
        df = df[['Position', 'N'] + [col for col in df.columns if col not in ['Position', 'N']]]

        def extract_sample_name(row):
            count_list = [i != 0 for i in row[2:]]
            samples = np.array(df.columns[2:])
            return ((',').join(samples[np.array(count_list)]))

        df['Samples'] = df.apply(extract_sample_name, axis=1)
        df = df[['Position', 'N', 'Samples'] + [col for col in df.columns if col not in ['Position', 'N', 'Samples']]]

    else:
        logger.info("This vcf file is not v4.2")
        sys.exit(1)

    return df


def import_VCF42_freebayes_to_tsv(vcf_file, sep='\t'):

    vcf_file = os.path.abspath(vcf_file)
    tsv_file = (".").join(vcf_file.split(".")[:-1]) + ".tsv"

    headers = []
    extra_fields = ['TYPE', 'DP', 'RO', 'AO']

    with open(tsv_file, 'w+') as fout:
        with open(vcf_file, 'r') as f:
            next_line = f.readline().strip()
            while next_line.startswith("#"):
                next_line = f.readline().strip()
                if next_line.startswith('#CHROM'):
                    headers = next_line.split('\t')

        headers = headers[:7] + extra_fields + ['OLDVAR']
        fout.write(("\t").join(headers) + "\n")

        with open(vcf_file, 'r') as f:
            for line in f:
                extra_field_list = []
                # and not 'complex' in line and not 'mnp' in line
                if not line.startswith("#"):
                    line_split = line.split(sep)[:8]
                    info = line_split[-1].split(";")
                    for field in extra_fields:
                        extra_field_list.append(
                            [x.split("=")[-1] for x in info if field in x][0])
                    if 'OLDVAR' in line:
                        extra_field_list.append([x.split("=")[-1]
                                                 for x in info if 'OLDVAR' in x][0].split(',')[0])
                    output_line = ("\t").join(
                        line_split[:7] + extra_field_list)
                    fout.write(output_line + "\n")


def import_tsv_freebayes_to_df(tsv_file, sep='\t'):

    tsv_file = os.path.abspath(tsv_file)
    df = pd.read_csv(tsv_file, sep=sep)

    df.rename(columns={'#CHROM': 'REGION', 'RO': 'REF_DP', 'DP': 'TOTAL_DP', 'AO': 'ALT_DP', 'QUAL': 'ALT_QUAL'}, inplace=True)

    df['REF_FREQ'] = df['REF_DP']/df['TOTAL_DP']
    df['ALT_FREQ'] = df['ALT_DP']/df['TOTAL_DP']

    df = df.sort_values(by=['POS']).reset_index(drop=True)

    return df[['REGION', 'POS', 'ID', 'REF', 'ALT', 'ALT_QUAL', 'FILTER', 'TOTAL_DP', 'TYPE', 'REF_DP', 'ALT_DP', 'REF_FREQ', 'ALT_FREQ', 'OLDVAR']]


def vcf_to_ivar_tsv(input_vcf, output_tsv):

    input_tsv = (".").join(input_vcf.split(".")[:-1]) + ".tsv"

    import_VCF42_freebayes_to_tsv(input_vcf)
    df = import_tsv_freebayes_to_df(input_tsv)

    df.to_csv(output_tsv, sep="\t", index=False)


# ── Bam variants ─────────────────────────────────────────────────────────

def run_snippy(r1, r2, reference, output_dir, sample, threads=16, minqual=20, minfrac=0.1, mincov=1):

    """
    snippy --cpus 16 --outdir mysnps --ref Listeria.gbk --R1 FDA_R1.fastq.gz --R2 FDA_R2.fastq.gz
    """

    prefix = os.path.join(output_dir, sample)

    cmd = ["snippy", "--cpus", str(threads), "--outdir", prefix, "--minqual", str(minqual), "--mincov", str(mincov), "--minfrac", str(minfrac), "--ref", reference, "--R1", r1, "--R2", r2]
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

    cmd = ["snippy-core", "-p", output_dir, "--ref", reference] + samples_snippy
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

    cmd = "samtools flagstat --threads {} {} > {}".format(str(threads), input_bam, output_file)
    execute_subprocess(cmd, isShell=True)


def create_coverage(input_bam, output_dir, sample):

    output_file = os.path.join(output_dir, sample + ".cov")

    cmd = "samtools depth -aa {} > {}".format(input_bam, output_file)
    execute_subprocess(cmd, isShell=True)


def add_window_distance(vcf_df, window_size=10):

    """
    Add a column indicating the maximum number of SNPs in a windows of 10 or supplied distance
    """

    list_pos = vcf_df.POS.to_list()  # all positions
    set_pos = set(list_pos)  # to set for later comparing
    max_pos = max(vcf_df.POS.to_list())

    all_list = list(range(1, max_pos + 1))  # create a list to slide one by one

    df_header = "window_" + str(window_size)

    vcf_df[df_header] = 1  # Create all 1 by default

    # Slide over windows
    for i in range(0, max_pos, 1):
        # This splits the list in windows of determined length
        window_pos = all_list[i:i+window_size]
        set_window_pos = set(window_pos)
        # How many known positions are in every window for later clasification
        num_conglomerate = set_pos & set_window_pos

        if len(num_conglomerate) > 1:
            for i in num_conglomerate:
                # Retrieve index with the known position
                index = vcf_df.index[vcf_df["POS"] == i][0]
                if vcf_df.loc[index, df_header] < len(num_conglomerate):
                    vcf_df.loc[index, df_header] = len(num_conglomerate)


def extract_close_snps(df, snps_in_10=1):

    # Calculate close SNPS/INDELS and remove those with 2 or more mutations in 10bp
    df['POS'] = df.apply(lambda x: x.Position.split('|')[2], axis=1)
    df['POS'] = df['POS'].astype(int)
    df = df.sort_values("POS")

    add_window_distance(df)

    return df[df.window_10 > snps_in_10].POS.tolist()


def coverage_to_df(input_file, min_coverage=10):

    sample_name = input_file.split("/")[-1].split(".")[0]
    min_cov_df = pd.DataFrame()
    coverage_list = []

    with open(input_file, 'r') as f:
        content = f.read()
        content_list = content.split('\n')
        while '' in content_list:
            content_list.remove('')

    coverage_list = [int(x.split("\t")[2]) for x in content_list]
    min_cov_df[sample_name] = coverage_list
    min_cov_df = min_cov_df[min_cov_df <= min_coverage].dropna(how='all')

    return min_cov_df


def identify_uncovered(cov_folder, min_coverage=10, nocall_fr=0.5):

    cov_folder = os.path.abspath(cov_folder)
    len_files = set()
    cov_df = pd.DataFrame()

    for root, _, files in os.walk(cov_folder):
        if root == cov_folder:
            for name in files:
                if name.endswith(".cov"):
                    filename = os.path.join(root, name)
                    # import to dataframe if they have the same positios(same reference)
                    low_coverage_df = coverage_to_df(filename)
                    cov_df = cov_df.merge(
                        low_coverage_df, how='outer', left_index=True, right_index=True)

    # Determine low covered positions in dataframe
    # Filter positions with values lower than min_cov, dro rows with all false and extract the indet to iterate
    df_any_uncovered = cov_df[cov_df < min_coverage].dropna(how='all')  # .index.tolist()
    df_any_uncovered['N_uncovered'] = df_any_uncovered.count(axis=1)
    df_any_uncovered['Position'] = df_any_uncovered.index + 1

    n_samples = len(df_any_uncovered.columns) - 2

    df_half_uncovered_list = df_any_uncovered['Position'][df_any_uncovered.N_uncovered /
                                                          n_samples >= nocall_fr].tolist()


def remove_position_from_compare(df, position_list):

    df['POS'] = df.apply(lambda x: x.Position.split('|')[2], axis=1)
    df['POS'] = df['POS'].astype(int)
    df = df.sort_values("POS")
    df = df[~df['POS'].isin(position_list)]
    df = df.drop(['POS'], axis=1)

    return df


# ── Species determination ────────────────────────────────────────────────

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


def kraken(r1_file, r2_file, report, kraken2_db, krona_html, threads=34):

    cmd_kraken = "kraken2 --db {} --memory-mapping --use-names --threads {} --report {} --gzip-compressed {} {} --output -".format(kraken2_db, str(threads), report, r1_file, r2_file)
    execute_subprocess(cmd_kraken, isShell=True)

    cmd_krona = "ktImportTaxonomy -m 3 -q 2 -t 5 {} -o {}".format(report, krona_html)
    execute_subprocess(cmd_krona, isShell=True)


# ── Annotation ───────────────────────────────────────────────────────────

def snpeff_execution(vcf_file, annot_file, database=False):

    with open(vcf_file, 'r') as f:
        content_list = f.read().split("\n")
        lines = len(content_list)

    if lines > 1:
        cmd = ["snpEff", "-noStats", database, vcf_file]
        with open(annot_file, "w+") as outfile:
            # calculate coverage and save it in th eoutput file
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    else:
        with open(annot_file, "w+") as outfile:
            outfile.write('No annotation found')


def import_annot_to_pandas(vcf_file, sep='\t'):

    """
    Order several annoattion by:
    Putative impact: Effects having higher putative impact are first.
    Effect type: Effects assumed to be more deleterious effects first.
    Canonical transcript before non-canonical.
    Marker genomic coordinates (e.g. genes starting before first)
    https://pcingola.github.io/SnpEff/se_inputoutput/
    Parse vcf outputted by snpEFF which adds the ANN field
    Dependences: calculate_ALT_AD
                calculate_true_ALT
    """

    header_lines = 0

    with open(vcf_file) as f:
        first_line = f.readline().strip()
        if first_line == 'No annotation found':
            return pd.read_csv(vcf_file, sep=sep)
        next_line = f.readline().strip()
        while next_line.startswith("##"):
            header_lines = header_lines + 1
            next_line = f.readline()

    # Use first line as header
    df = pd.read_csv(vcf_file, sep=sep, skiprows=[header_lines], header=header_lines)

    ann_headers = ['Allele',
                   'Annotation',
                   'Annotation_Impact',
                   'Gene_Name',
                   'Gene_ID',
                   'Feature_Type',
                   'Feature_ID',
                   'Transcript_BioType',
                   'Rank',
                   'HGVS.c',
                   'HGVS.p',
                   'cDNA.pos / cDNA.length',
                   'CDS.pos / CDS.length',
                   'AA.pos / AA.length',
                   'ERRORS / WARNINGS / INFO']

    df['Type'] = df.apply(lambda row: 'snp' if len(row['REF']) == 1 and len(row['ALT']) == 1 else 'indel', axis=1)

    anlelle_headers = ['DP', 'REF_DP', 'ALT_DP']
    df[anlelle_headers] = pd.DataFrame([[None, None, None]] * len(df), columns=anlelle_headers)

    def assign_values(row):
        info_parts = row.INFO.split(';') if isinstance(row.INFO, str) else []

        def get_value(prefix):
            for part in info_parts:
                if part.startswith(prefix):
                    return part.split('=')[1]
            return None

        if row['Type'] == 'snp':
            values = [get_value('DP='), get_value('RO='), get_value('AO=')]
        elif row['Type'] in ['indel', 'del', 'ins']:
            values = [get_value('DP='), get_value('RO='), get_value('AO=')]
        else:
            values = [None, None, None]

        return pd.Series(values)

    df[anlelle_headers] = df.apply(assign_values, axis=1)

    for head in anlelle_headers:
        df[head] = df[head].str.split("=").str[-1]

    df['ALT_FREQ'] = pd.to_numeric(df['ALT_DP']) / pd.to_numeric(df['DP'])

    df['TMP_ANN_16'] = df['INFO'].apply(lambda x: ('|').join(x.split('|')[0:15]))

    df.INFO = df.INFO.str.split("ANN=").str[-1]

    df = df.join(df.pop('INFO')
                   .str.strip(',')
                   .str.split(',', expand=True)
                   .stack()
                   .reset_index(level=1, drop=True)
                   .rename('INFO')).reset_index(drop=True)

    df['TMP_ANN_16'] = df['INFO'].apply(lambda x: ('|').join(x.split('|')[0:15]))
    df[ann_headers] = df['TMP_ANN_16'].str.split('|', expand=True)
    df['HGVS.c'] = df['HGVS.c'].str.split(".").str[-1]
    df['HGVS.p'] = df['HGVS.p'].str.split(".").str[-1].replace('', '-')

    df.drop(["INFO", "TMP_ANN_16", "REF_DP", "ALT_DP", "Allele", "ERRORS / WARNINGS / INFO", "cDNA.pos / cDNA.length"], inplace=True, axis=1)

    return df


def annotate_snpeff(input_vcf_file, output_annot_file, database='NC_045512.2'):

    snpeff_execution(input_vcf_file, output_annot_file, database=database)

    annot_df = import_annot_to_pandas(output_annot_file)
    annot_df.to_csv(output_annot_file, sep="\t", index=False)


def rename_reference_snpeff(input_vcf, output_vcf, new_ref_name='Chromosome'): 

    """
    If you have to create a custom snpeff database, you should maybe change the ref_name to annotate or build the custom database changing de genome ID by Chromosome
    https://pcingola.github.io/SnpEff/snpeff/build_db/
    """

    with open(input_vcf) as f:
        next_line = f.readline().strip()
        while next_line.startswith("##"):
            next_line = f.readline()
        old_ref = f.readline().split("\t")[0].strip()

    with open(output_vcf, 'w+') as fo:
        with open(input_vcf, 'r') as f:
            content = f.read()
            new_content = content.replace(old_ref, new_ref_name)
            fo.write(new_content)


def bed_to_df(bed_file):

    """
    Import bed file separated by tabs into a pandas df
    -Handle header line
    -Handle with and without description (If there is no description adds true or false to annotated df)
    """

    header_lines = 0
    # Handle likely header by checking colums 2 and 3 as numbers
    with open(bed_file, 'r') as f:
        next_line = f.readline().strip()
        line_split = next_line.split(None)  # This split by any blank character
        start = line_split[1]
        end = line_split[2]
        while not start.isdigit() and not end.isdigit():
            header_lines = header_lines + 1
            next_line = f.readline().strip()
            # This split by any blank character
            line_split = next_line.split(None)
            start = line_split[1]
            end = line_split[2]

    if header_lines == 0:
        df = pd.read_csv(bed_file, sep="\t", header=None)
    else:
        df = pd.read_csv(bed_file, sep="\t", skiprows=header_lines, header=None)

    df = df.iloc[:, 0:4]
    df.columns = ["#CHROM", "start", "end", "description"]

    return df


def add_bed_info(bed_df, position):

    """
    Identify a position within a range
    credits: https://stackoverflow.com/questions/6053974/python-efficiently-check-if-integer-is-within-many-ranges
    """

    if any(start <= position <= end for (start, end) in zip(bed_df.start.values.tolist(), bed_df.end.values.tolist())):
        description_out = bed_df.description[(bed_df.start <= position) & (bed_df.end >= position)].values[0]
        return description_out
    else:
        return None


def annotate_bed_s(tsv_df, bed_files):

    with open(tsv_df, 'r') as f:
        content = f.read().strip()
        if content == 'No annotation found':
            return pd.DataFrame(columns=['POS', 'REF', 'ALT', 'INFO'])

        else:
            df = pd.read_csv(tsv_df, sep="\t")

            variable_list = [x.split("/")[-1].split(".")[0] for x in bed_files]

            for variable_name, bed_file in zip(variable_list, bed_files):
                logger.info("ANNOTATING BED: {}".format(bed_file))
                bed_annot_df = bed_to_df(bed_file)
                df[variable_name] = df['POS'].apply(lambda x: add_bed_info(bed_annot_df, x))
            return df


def import_VCF_to_pandas(vcf_file):

    header_lines = 0

    with open(vcf_file) as f:
        first_line = f.readline().strip()
        next_line = f.readline().strip()
        while next_line.startswith("##"):
            header_lines = header_lines + 1
            next_line = f.readline()

    if first_line.startswith('##'):
        df = pd.read_csv(vcf_file, sep='\t', skiprows=[header_lines], header=header_lines)

        df['ALT'] = df['ALT'].str.upper()
        df['REF'] = df['REF'].str.upper()

        if 'INFO' in df.columns:
            return df
        else:
            last_column = df.columns[-1]
            df = df.rename(columns={last_column: 'INFO'})
            return df
    else:
        logger.info("This vcf file is not properly formatted")
        sys.exit(1)


def annotate_vcfs(tsv_df, vcfs, include_ref=False):

    df = pd.read_csv(tsv_df, sep="\t")

    for vcf in vcfs:
        logger.info("ANNOTATING VCF: {}".format(vcf))
        header = (".").join(vcf.split("/")[-1].split(".")[0:-1])
        dfvcf = import_VCF_to_pandas(vcf)

        if include_ref:
            dfvcf = dfvcf[['POS', 'REF', 'ALT', 'INFO']]
        else:
            dfvcf = dfvcf[['POS', 'ALT', 'INFO']]
        dfvcf = dfvcf.rename(columns={'INFO': header})
        df = df.merge(dfvcf, how='left')

    return df


def user_annotation(tsv_file, output_file, vcf_files=[], bed_files=[]):

    bed_df = annotate_bed_s(tsv_file, bed_files)
    vcf_df = annotate_vcfs(tsv_file, vcf_files)

    df = bed_df.merge(vcf_df)
    df.to_csv(output_file, sep="\t", index=False)


def checkAA(snpEffRow, dfAnnot, gene):

    df = dfAnnot
    df['aaAnnot'] = df['aa'] + ":" + df['annot']
    presence_list = [annot in snpEffRow for annot in dfAnnot.aa]

    if any(":" in annot for annot in dfAnnot.annot):
        for idx, row in dfAnnot.iterrows():
            if ":" in row.annot:
                annot_split = row.annot.split(":")
                if row.aa in snpEffRow and annot_split[0] in gene:
                    return annot_split[-1]
    else:
        annotation_list = np.array(df.aaAnnot.tolist())
        return (',').join(annotation_list[np.array(presence_list)])


def annotate_aas(annot_file, aas):

    df = pd.read_csv(annot_file, sep="\t")
    df = df.drop_duplicates(subset=["POS", "ALT"], keep="first")

    for aa in aas:
        header = (".").join(aa.split("/")[-1].split(".")[0:-1])
        dfaa = pd.read_csv(aa, sep="\t", names=['aa', 'annot'])
        if not header in df.columns:
            print("ANNOTATING AA: {}".format(aa))
            df['HGVS.p'] = df['HGVS.p'].astype(str)
            df[header] = df.apply(lambda x: checkAA(
                x['HGVS.p'], dfaa, x['Gene_Name']), axis=1)
        else:
            print("SKIPPED AA: {}".format(aa))

    return df


def user_annotation_aa(annot_file, output_file, aa_files=[]):

    with open(annot_file, 'r') as f:
        content = f.read().strip()
        if content == 'No annotation found':
            logger.debug("{} file has NO Annotation".format(annot_file))
            with open(output_file, 'w+') as fout:
                fout.write('No annotation found')
        else:
            df = annotate_aas(annot_file, aa_files)
            df.to_csv(output_file, sep="\t", index=False)
