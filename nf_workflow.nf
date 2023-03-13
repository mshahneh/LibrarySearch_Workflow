#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputlibraries = "data/libraries"
params.inputspectra = "data/spectra"

TOOL_FOLDER = "$baseDir/bin"

process searchData {
    publishDir "./nf_output", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    each file(input_library)
    each file(input_spectrum)

    //output:
    //file 'output.tsv'

    """
    mkdir search_results
    python $TOOL_FOLDER/library_search_wrapper.py \
    $input_library $input_spectrum search_results \
    $TOOL_FOLDER/convert \
    $TOOL_FOLDER/main_execmodule
    """
}

workflow {
    libraries = Channel.fromPath(params.inputlibraries + "/*.mgf" )
    spectra = Channel.fromPath(params.inputspectra + "/*" )
    searchData(libraries, spectra)
}