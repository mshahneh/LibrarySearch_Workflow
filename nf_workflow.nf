#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputlibraries = "data/libraries"
params.inputspectra = "data/spectra"

// Parameters
params.topk = 1

params.ion_tolerance = 0.5
params.pm_tolerance = 2.0

params.cosine_threshold = 0.7
params.min_match = 6

params.filter_precursor = 1
params.filter_window = 1

TOOL_FOLDER = "$baseDir/bin"

process searchData {
    //publishDir "./nf_output", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    each file(input_library)
    each file(input_spectrum)

    output:
    file 'search_results/*' optional true

    """
    mkdir search_results
    python $TOOL_FOLDER/library_search_wrapper.py \
    $input_spectrum $input_library search_results \
    $TOOL_FOLDER/convert \
    $TOOL_FOLDER/main_execmodule.allcandidates
    """
}

process mergeResults {
    publishDir "./nf_output", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path "results/*"

    output:
    path 'merged_results.tsv'

    """
    python $TOOL_FOLDER/tsv_merger.py \
    results merged_results.tsv
    """
}

process getGNPSAnnotations {
    publishDir "./nf_output", mode: 'copy'

    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path "merged_results.tsv"

    output:
    path 'merged_results_with_gnps.tsv'

    """
    python $TOOL_FOLDER/getGNPS_library_annotations.py \
    merged_results.tsv merged_results_with_gnps.tsv
    """

}

workflow {
    libraries = Channel.fromPath(params.inputlibraries + "/*.mgf" )
    spectra = Channel.fromPath(params.inputspectra + "/*" )
    
    search_results = searchData(libraries, spectra)

    // TODO: We'll want to collate them into batches and then batch the batches
    merged_results = mergeResults(search_results.collect())

    getGNPSAnnotations(merged_results)
}