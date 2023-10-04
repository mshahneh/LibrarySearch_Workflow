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
params.analog_search = "0"
params.analog_max_shift = 1999

// Blink Parameters
params.blink_ionization = "positive"
params.blink_minpredict = 0.01

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
    --library_min_matched_peaks $params.library_min_matched_peaks \
    --analog_search $params.analog_search
    """
}

process searchDataBlink {
    //publishDir "./nf_output", mode: 'copy'

    conda "$TOOL_FOLDER/blink/environment.yml"

    input:
    each file(input_library)
    each file(input_spectrum)

    output:
    file 'search_results/*.csv' optional true

    script:
    def randomFilename = UUID.randomUUID().toString()
    def input_spectrum_abs = input_spectrum.toRealPath()
    def input_library_abs = input_library.toRealPath()
    """
    mkdir search_results
    echo $workDir
    previous_cwd=\$(pwd)
    echo \$previous_cwd

    cd $TOOL_FOLDER/blink && python -m blink.blink_cli \
    $input_spectrum_abs \
    $input_library_abs \
    \$previous_cwd/search_results/${randomFilename}.csv \
    $TOOL_FOLDER/blink/models/positive_random_forest.pickle \
    $TOOL_FOLDER/blink/models/negative_random_forest.pickle \
    $params.blink_ionization \
    --min_predict $params.blink_minpredict \
    --mass_diffs 0 14.0157 12.000 15.9949 2.01565 27.9949 26.0157 18.0106 30.0106 42.0106 1.9792 17.00284 24.000 13.97925 1.00794 40.0313 \
    --tolerance $params.fragment_tolerance
    """
}

process formatBlinkResults {
    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path input_file

    output:
    path '*.tsv'

    """
    python $TOOL_FOLDER/format_blink.py \
    $input_file \
    ${input_file}.tsv
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
        search_results = searchDataGNPS(libraries, spectra)

        // TODO: We'll want to collate them into batches and then batch the batches
        merged_results = mergeResults(search_results.collect())
    }
    else if (params.searchtool == "blink"){
        search_results = searchDataBlink(libraries, spectra)

        formatted_results = formatBlinkResults(search_results)

        merged_results = mergeResults(formatted_results.collect())
    }

    getGNPSAnnotations(merged_results)
}