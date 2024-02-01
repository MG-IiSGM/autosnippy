#!/usr/bin/env python

import os
import sys
import logging
import pandas as pd
import numpy as np
import re
import subprocess
from tabulate import tabulate
from misc import check_create_dir, execute_subprocess

logger = logging.getLogger()


# Import files containing annotation info and convert them to dictionary
#script_dir = os.path.dirname(os.path.realpath(__file__))

def make_blast(query_fasta, database, sample, output_folder, db_type="nucl", query_type="nucl", evalue=0.0001, threads=8):

    blast_command = 'blastn' if query_type == "nucl" else 'blastp'
    database_name = database.split("/")[-1].split(".")[0]
    output_database_tmp = os.path.join(
        output_folder, database_name + ".blast.tmp")
    output_blast = os.path.join(
        output_folder, sample + "." + database_name + ".blast")
    blastdb_cmd = ['makeblastdb', '-in', database,
                   '-out', output_database_tmp, '-dbtype', db_type]

    logger.info((',').join(blastdb_cmd))

    execute_subprocess(blastdb_cmd)

    blast_cmd = [blast_command, "-query", query_fasta, "-db", output_database_tmp,
                 "-out", output_blast, "-evalue", str(
                     evalue), "-num_threads", str(threads),
                 "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"]

    logger.info((',').join(blast_cmd))

    execute_subprocess(blast_cmd)


def tsv_to_vcf(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t")
    is_empty = df.shape[0] == 0
    #df.insert(2, 'ID', '.')
    df.fillna(".", inplace=True)
    df["PASS"].replace({True: 'PASS'}, inplace=True)
    df.rename(columns={"REGION": "#CHROM", "GFF_FEATURE": "ID",
                       "ALT_QUAL": "QUAL", "PASS": "FILTER"}, inplace=True)

    fial_columns = ['#CHROM', 'POS', 'ID',
                    'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']

    if not is_empty:
        df['INFO'] = df.apply(lambda x: "CODON={}-{};AA={}-{};DP={};ALT_FREQ={:.2f}".format(
            x.REF_CODON, x.ALT_CODON, x.REF_AA, x.ALT_AA, x.TOTAL_DP, x.ALT_FREQ), axis=1)
    else:
        df = df.reindex(columns=fial_columns)
    df = df[fial_columns]

    return df


def rename_reference_snpeff(input_vcf, output_vcf, new_ref_name='Chromosome'): # if you have to create a custom snpeff database, you should maybe change the ref_name to annotate or build the custom database changing de genome ID by Chromosome
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


def snpeff_execution(vcf_file, annot_file, database=False):
    with open(vcf_file, 'r') as f:
        content_list = f.read().split("\n")
        lines = len(content_list)

    if lines > 1:
        cmd = ["snpEff", "-noStats", database, vcf_file]
        with open(annot_file, "w+") as outfile:
            # calculate coverage and save it in th eoutput file
            subprocess.run(cmd,
                           stdout=outfile, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    else:
        with open(annot_file, "w+") as outfile:
            outfile.write('No annotation found')


# def import_annot_to_pandas(vcf_file, sep='\t'):
#     """
#     Order several annoattion by:
#     Putative impact: Effects having higher putative impact are first.
#     Effect type: Effects assumed to be more deleterious effects first.
#     Canonical transcript before non-canonical.
#     Marker genomic coordinates (e.g. genes starting before first)
#     https://pcingola.github.io/SnpEff/se_inputoutput/
#     Parse vcf outputted by snpEFF which adds the ANN field
#     Dependences: calculate_ALT_AD
#                 calculate_true_ALT
#     """
#     header_lines = 0
#     with open(vcf_file) as f:
#         first_line = f.readline().strip()
#         if first_line == 'No annotation found':
#             return pd.read_csv(vcf_file, sep=sep)
#         next_line = f.readline().strip()
#         while next_line.startswith("##"):
#             header_lines = header_lines + 1
#             # logger.info(next_line)
#             next_line = f.readline()

#     # Use first line as header
#     df = pd.read_csv(vcf_file, sep=sep, skiprows=[
#                      header_lines], header=header_lines)

#     ann_headers = ['Allele',
#                    'Annotation',
#                    'Annotation_Impact',
#                    'Gene_Name',
#                    'Gene_ID',
#                    'Feature_Type',
#                    'Feature_ID',
#                    'Transcript_BioType',
#                    'Rank',
#                    'HGVS.c',
#                    'HGVS.p',
#                    'cDNA.pos / cDNA.length',
#                    'CDS.pos / CDS.length',
#                    'AA.pos / AA.length',
#                    'ERRORS / WARNINGS / INFO']
#     anlelle_headers = ['Codon_change', 'AA_change', 'DP', 'ALT_FREQ']

#     # Apply function to split and recover the first 15 fields = only first anotations, the most likely

#     df[anlelle_headers] = df.apply(lambda x: x.INFO.split(';')[
#                                    0:4], axis=1, result_type="expand")

#     for head in anlelle_headers:
#         df[head] = df[head].str.split("=").str[-1]

#     df['TMP_ANN_16'] = df['INFO'].apply(
#         lambda x: ('|').join(x.split('|')[0:15]))

#     df.INFO = df.INFO.str.split("ANN=").str[-1]

#     df = df.join(df.pop('INFO')
#                    .str.strip(',')
#                    .str.split(',', expand=True)
#                    .stack()
#                    .reset_index(level=1, drop=True)
#                    .rename('INFO')).reset_index(drop=True)

#     df['TMP_ANN_16'] = df['INFO'].apply(
#         lambda x: ('|').join(x.split('|')[0:15]))
#     df[ann_headers] = df['TMP_ANN_16'].str.split('|', expand=True)
#     df['HGVS.c'] = df['HGVS.c'].str.split(".").str[-1]
#     df['HGVS.p'] = df['HGVS.p'].str.split(".").str[-1].replace('', '-')

#     df.drop(["INFO", "TMP_ANN_16"], inplace=True, axis=1)

#     return df


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
            # logger.info(next_line)
            next_line = f.readline()

    # Use first line as header
    df = pd.read_csv(vcf_file, sep=sep, skiprows=[
                     header_lines], header=header_lines)

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
        if row['Type'] == 'snp':
            values = row.INFO.split(';')[1:4] if isinstance(row.INFO, str) else [None, None, None]
        elif row['Type'] == 'indel':
            values = [
                row.INFO.split(';')[2] if isinstance(row.INFO, str) else None,
                row.INFO.split(';')[5] if isinstance(row.INFO, str) else None,
                row.INFO.split(';')[1] if isinstance(row.INFO, str) else None
            ]
        else:
            values = [None, None, None]
        return pd.Series(values)

    df[anlelle_headers] = df.apply(assign_values, axis=1)

    for head in anlelle_headers:
        df[head] = df[head].str.split("=").str[-1]

    df['ALT_FREQ'] = pd.to_numeric(df['ALT_DP']) / pd.to_numeric(df['DP'])

    # return df

    df['TMP_ANN_16'] = df['INFO'].apply(
        lambda x: ('|').join(x.split('|')[0:15]))

    df.INFO = df.INFO.str.split("ANN=").str[-1]

    df = df.join(df.pop('INFO')
                   .str.strip(',')
                   .str.split(',', expand=True)
                   .stack()
                   .reset_index(level=1, drop=True)
                   .rename('INFO')).reset_index(drop=True)

    df['TMP_ANN_16'] = df['INFO'].apply(
        lambda x: ('|').join(x.split('|')[0:15]))
    df[ann_headers] = df['TMP_ANN_16'].str.split('|', expand=True)
    df['HGVS.c'] = df['HGVS.c'].str.split(".").str[-1]
    df['HGVS.p'] = df['HGVS.p'].str.split(".").str[-1].replace('', '-')

    df.drop(["INFO", "TMP_ANN_16", "REF_DP", "ALT_DP", "Allele", "ERRORS / WARNINGS / INFO", "cDNA.pos / cDNA.length"], inplace=True, axis=1)

    return df


def annotate_snpeff(input_vcf_file, output_annot_file, database='NC_045512.2'):
    # Execure snpEff
    snpeff_execution(input_vcf_file, output_annot_file, database=database)
    # Format annot vcf and remove vcf
    annot_df = import_annot_to_pandas(output_annot_file)
    annot_df.to_csv(output_annot_file, sep="\t", index=False)


def annotate_pangolin(input_file, output_folder, output_filename, threads=8, max_ambig=0.6):
    cmd = ["pangolin", input_file, "--outdir", output_folder, "--outfile",
           output_filename, "--threads", str(threads), "--max-ambig", str(max_ambig)]
    execute_subprocess(cmd)


def get_reverse(nucleotyde):
    nucleotyde = str(nucleotyde)
    nucleotyde_rev = {'A': 'T',
                      'T': 'A',
                      'C': 'G',
                      'G': 'C'}
    if len(nucleotyde) > 1:
        nucleotyde_str = nucleotyde[::-1]  # Reverse nucleotide
        nucleotyde_str_fin = "".join(
            [nucleotyde_rev[x] for x in nucleotyde_str])  # Complement nucleotide
        return nucleotyde_str_fin
    else:
        return nucleotyde_rev[nucleotyde]


def import_VCF_to_pandas(vcf_file):
    header_lines = 0
    with open(vcf_file) as f:
        first_line = f.readline().strip()
        next_line = f.readline().strip()
        while next_line.startswith("##"):
            header_lines = header_lines + 1
            # logger.info(next_line)
            next_line = f.readline()

    if first_line.startswith('##'):
        df = pd.read_csv(vcf_file, sep='\t', skiprows=[
                         header_lines], header=header_lines)

        df['ALT'] = df['ALT'].str.upper()
        df['REF'] = df['REF'].str.upper()
        # Check INFO
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
        # delim_whitespace=True
        df = pd.read_csv(bed_file, sep="\t", header=None)
    else:
        df = pd.read_csv(bed_file, sep="\t", skiprows=header_lines,
                         header=None)  # delim_whitespace=True

    df = df.iloc[:, 0:4]
    df.columns = ["#CHROM", "start", "end", "description"]

    return df


def add_bed_info(bed_df, position):
    """
    Identify a position within a range
    credits: https://stackoverflow.com/questions/6053974/python-efficiently-check-if-integer-is-within-many-ranges
    """
    #dict_position = bed_to_dict(bed_file)
    if any(start <= position <= end for (start, end) in zip(bed_df.start.values.tolist(), bed_df.end.values.tolist())):
        description_out = bed_df.description[(
            bed_df.start <= position) & (bed_df.end >= position)].values[0]
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

            # extract file name and use it as header
            variable_list = [x.split("/")[-1].split(".")[0] for x in bed_files]

            for variable_name, bed_file in zip(variable_list, bed_files):
                logger.info("ANNOTATING BED: {}".format(bed_file))
                bed_annot_df = bed_to_df(bed_file)
                df[variable_name] = df['POS'].apply(
                    lambda x: add_bed_info(bed_annot_df, x))
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
                    return row.aaAnnot
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
            # Filtra la salida de SNPEff con las anotaciones aa
            # df.drop_duplicates(subset=['HGVS.p'], keep='first', inplace=True)
            df.to_csv(output_file, sep="\t", index=False)


html_template = """
<!DOCTYPE html>
<html>
    
  <head>
    <script src="http://code.jquery.com/jquery-3.3.1.min.js"></script>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet" type="text/css" />
    <link href="https://nightly.datatables.net/css/dataTables.bootstrap4.css" rel="stylesheet" type="text/css" />


    <script src="https://nightly.datatables.net/js/jquery.dataTables.js"></script>
    <script src="https://nightly.datatables.net/js/dataTables.bootstrap4.js"></script>

    <style>
        body {
        font: 90%/1rem "Helvetica Neue", HelveticaNeue, Verdana, Arial, Helvetica, sans-serif;
        margin: 0;
        padding: 0;
        color: #333;
        background-color: #fff;
        }
    </style>

    
    <meta charset=utf-8 />
    <title>COVID Variant report</title>
    <meta name="description" content="https://github.com/pedroscampoy/covid_multianalysis">
    <meta name="author" content="pedroscampoy@gmail.com">
  </head>
  <body>
    <div class="container-fluid">
        TABLESUMMARY
    </div>

    <script>
        $(document).ready( function () {
            var table = $('#variants').DataTable({
                orderCellsTop: true,
                initComplete: function () {
                    this.api().columns([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).every( function () {
                        var column = this;
                        var select = $('<select><option value=""></option></select>')
                        .appendTo(  $('thead tr:eq(1) th:eq(' + this.index()  + ')') )
                            .on( 'change', function () {
                                var val = $.fn.dataTable.util.escapeRegex(
                                    $(this).val()
                                );
                                column
                                    .search( val ? '^'+val+'$' : '', true, false )
                                    .draw();
                            } );
        
                        column.data().unique().sort().each( function ( d, j ) {
                            select.append( '<option value="'+d+'">'+d+'</option>' )
                        } );
                    } );
                }
            });
        } );
    </script>
  </body>
</html> 
"""

report_samples_html = """
<!DOCTYPE html>
<html>
  <head>
    <script src="http://code.jquery.com/jquery-3.3.1.min.js"></script>

    <link
      href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css"
      rel="stylesheet"
      type="text/css"
    />
    <link
      href="https://nightly.datatables.net/css/dataTables.bootstrap4.css"
      rel="stylesheet"
      type="text/css"
    />

    <script src="https://nightly.datatables.net/js/jquery.dataTables.js"></script>
    <script src="https://nightly.datatables.net/js/dataTables.bootstrap4.js"></script>

    <style>
        html {
        height: 100%;
        }
      body {
        font: 90%/1rem "Helvetica Neue", HelveticaNeue, Verdana, Arial,
          Helvetica, sans-serif;
        margin: 0;
        padding: 0;
        color: #333;
        background-color: #fff;
        height: 100%;
      }

      .dropdown {
        margin: 20px;
      }

      .dropdown-menu {
        max-height: 20rem;
        overflow-y: auto;
      }

      object {
        width: 100%;
        height: 100%;
      }
    </style>

    <meta charset="utf-8" />
    <title>COVID Variant report</title>
    <meta
      name="description"
      content="https://github.com/pedroscampoy/covid_multianalysis"
    />
    <meta name="author" content="pedroscampoy@gmail.com" />
  </head>
  <body>
    <div class="dropdown">
      <button
        class="btn btn-secondary dropdown-toggle"
        type="button"
        id="dropdown_samples"
        data-toggle="dropdown"
        aria-haspopup="true"
        aria-expanded="false"
      >
        Sample
      </button>
      <div id="menu" class="dropdown-menu" aria-labelledby="dropdown_samples">
        <form class="px-4 py-2">
          <input
            type="search"
            class="form-control"
            id="searchSample"
            placeholder="20000000"
            autofocus="autofocus"
          />
        </form>
        <div id="menuItems"></div>
        <div id="empty" class="dropdown-header">No samples found</div>
      </div>
    </div>

    <div class="container-fluid w-100 h-100 mh-100" id="display-table">
      
    </div>

    <script>
      $(document).ready(function () {
        var table = $("#variants").DataTable({
          orderCellsTop: true,
          initComplete: function () {
            this.api()
              .columns([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
              .every(function () {
                var column = this;
                var select = $('<select><option value=""></option></select>')
                  .appendTo($("thead tr:eq(1) th:eq(" + this.index() + ")"))
                  .on("change", function () {
                    var val = $.fn.dataTable.util.escapeRegex($(this).val());
                    column
                      .search(val ? "^" + val + "$" : "", true, false)
                      .draw();
                  });

                column
                  .data()
                  .unique()
                  .sort()
                  .each(function (d, j) {
                    select.append(
                      '<option value="' + d + '">' + d + "</option>"
                    );
                  });
              });
          },
        });
      });

      //https://stackoverflow.com/questions/45007712/bootstrap-4-dropdown-with-search
      //Initialize with the list of symbols
      let names = ["ALLSAMPLES"];

      //Find the input search box
      let search = document.getElementById("searchSample");

      //Find every item inside the dropdown
      let items = document.getElementsByClassName("dropdown-item");

      buildDropDown = (values) => {
        let contents = [];
        for (let name of values) {
          contents.push(
            '<input type="button" class="dropdown-item" type="button" value="' +
              name +
              '"/>'
          );
        }
        $("#menuItems").append(contents.join(""));

        //Hide the row that shows no items were found
        $("#empty").hide();
      }

      //Capture the event when user types into the search box
      window.addEventListener("input", () => filter(search.value.trim().toLowerCase()));

      //For every word entered by the user, check if the symbol starts with that word
      //If it does show the symbol, else hide it
      function filter(word) {
        let length = items.length;
        let collection = [];
        let hidden = 0;
        for (let i = 0; i < length; i++) {
          if (items[i].value.toLowerCase().includes(word)) {
            $(items[i]).show();
          } else {
            $(items[i]).hide();
            hidden++;
          }
        }

        //If all items are hidden, show the empty view
        if (hidden === length) {
          $("#empty").show();
        } else {
          $("#empty").hide();
        }
      }

      //If the user clicks on any item, set the title of the button as the text of the item
      $("#menuItems").on("click", ".dropdown-item", function () {
        $("#dropdown_samples").text($(this)[0].value);
        $("#dropdown_samples").dropdown("toggle");
        document.getElementById("display-table").innerHTML=`<object type="text/html" data="${$(this)[0].value}.html" ></object>`;

      });

      buildDropDown(names);
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.0.0-beta.2/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""


# def annotation_to_html(file_annot, sample):
#     folder = ('/').join(file_annot.split('/')[0:-1])

#     logger.debug('Adapting html in sample: {}'.format(sample))

#     df = pd.read_csv(file_annot, sep="\t", dtype=str)
#     df['ALT_FREQ'] = df['ALT_FREQ'].astype(float)
#     df['POS'] = df['POS'].astype(int)

#     logger.debug('read csv {}'.format(file_annot))

#     #dtype={"user_id": int, "username": "string"}

#     df = df[['#CHROM', 'POS', 'REF', 'ALT', 'Codon_change',
#              'AA_change', 'DP', 'ALT_FREQ', 'Annotation',
#              'Annotation_Impact', 'Gene_Name', 'HGVS.p'] + df.columns[26:].tolist()]
#     if 'Variants' in df.columns:
#         df = df.drop('Variants', axis=1)
#     if 'DVariant' in df.columns:
#         df = df.drop('DVariant', axis=1)
#     df = df.drop_duplicates(
#         subset=['#CHROM', 'POS', 'REF', 'ALT'], keep="first")
#     df = df[df.ALT_FREQ >= 0.2]

#     def handle_aa(x): return None if x != x else x.split(':')[1]
#     df.iloc[:, 12:] = df.iloc[:, 12:].applymap(handle_aa)

#     df = pd.melt(df, id_vars=['#CHROM', 'POS', 'REF', 'ALT', 'Codon_change', 'AA_change', 'DP',
#                               'ALT_FREQ', 'Annotation', 'Annotation_Impact', 'Gene_Name', 'HGVS.p'], value_vars=df.columns[12:].tolist())

#     if 'variable' in df.columns:
#         df = df.drop('variable', axis=1)
#     df = df.rename(columns={'value': 'variable'})

#     table = tabulate(df, headers='keys', tablefmt='html', showindex=False)
#     table = table.replace(
#         "<table>", "<table id=\"variants\" class=\"table table-striped table-bordered nowrap\" width=\"100%\">")
#     table = table.replace("style=\"text-align: right;\"", "")

#     row_filter = "<tr>\n" + "<th></th>\n" * len(df.columns) + "</tr>\n"

#     table = table.replace("</tr>\n</thead>", "</tr>\n" +
#                           row_filter + "</thead>")

#     final_html = html_template.replace('TABLESUMMARY', table)

#     with open(os.path.join(folder, sample + ".html"), 'w+') as f:
#         f.write(final_html)


if __name__ == '__main__':
    logger.info("#################### ANNOTATION #########################")
