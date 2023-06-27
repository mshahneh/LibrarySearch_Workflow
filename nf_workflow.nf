#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputlibraries = "data/libraries"
params.inputspectra = "data/spectra"

// Parameters
params.searchtool = "gnps" // blink

params.topk = 1

params.fragment_tolerance = 0.5
params.pm_tolerance = 2.0

params.library_min_cosine = 0.7
params.library_min_matched_peaks = 6

//TODO: Implement This
params.filter_precursor = 1
params.filter_window = 1

//TODO: Implement This
params.analog_search = 0
params.analog_max_shift = 1999

TOOL_FOLDER = "$baseDir/bin"

process searchDataGNPS {
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
    $TOOL_FOLDER/main_execmodule.allcandidates \
    --pm_tolerance $params.pm_tolerance \
    --fragment_tolerance $params.fragment_tolerance \
    --topk $params.topk \
    --library_min_cosine $params.library_min_cosine \
    --library_min_matched_peaks $params.library_min_matched_peaks
    """
}

process searchDataBlink {
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
    $TOOL_FOLDER/main_execmodule.allcandidates \
    --pm_tolerance $params.pm_tolerance \
    --fragment_tolerance $params.fragment_tolerance \
    --topk $params.topk \
    --library_min_cosine $params.library_min_cosine \
    --library_min_matched_peaks $params.library_min_matched_peaks
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
    results \
    merged_results.tsv \
    --topk $params.topk
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
    merged_results.tsv \
    merged_results_with_gnps.tsv
    """

}

workflow {
    libraries = Channel.fromPath(params.inputlibraries + "/*.mgf" )
    spectra = Channel.fromPath(params.inputspectra + "/**" )
    
    if(params.searchtool == "gnps"){
        search_results = searchData(libraries, spectra)

        // TODO: We'll want to collate them into batches and then batch the batches
        merged_results = mergeResults(search_results.collect())
    }
    else if (params.searchtool == "blink"){
    }

    getGNPSAnnotations(merged_results)
}