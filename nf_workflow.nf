#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputlibraries = "data/libraries"
params.inputspectra = "data/spectra"

// Parameters
params.searchtool = "gnps" // blink, gnps

params.topk = 1

params.fragment_tolerance = 0.5
params.pm_tolerance = 2.0

params.library_min_cosine = 0.7
params.library_min_matched_peaks = 6

params.merge_batch_size = 1000 //Not a UI parameter

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
    tuple file(input_library), file(input_spectrum), val(input_path), val(full_path)

    output:
    file 'search_results/*' optional true

    """
    mkdir search_results

    python $TOOL_FOLDER/library_search_wrapper.py \
        "$input_spectrum" \
        "$input_library" \
        search_results \
        $TOOL_FOLDER/convert \
        $TOOL_FOLDER/main_execmodule.allcandidates \
        --pm_tolerance "$params.pm_tolerance" \
        --fragment_tolerance "$params.fragment_tolerance" \
        --topk $params.topk \
        --library_min_cosine $params.library_min_cosine \
        --library_min_matched_peaks $params.library_min_matched_peaks \
        --analog_search "$params.analog_search" \
        --full_relative_query_path "$full_path"
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

process chunkResults {
    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path to_merge, stageAs: './results/*' // To avoid naming collisions

    output:
    path "batched_results.tsv" optional true

    """

    python $TOOL_FOLDER/tsv_merger.py \
    results \
    batched_results.tsv \
    --topk $params.topk
    """
}

// Use a separate process to merge all the batched results
process mergeResults {
    conda "$TOOL_FOLDER/conda_env.yml"

    input:
    path 'batched_results.tsv', stageAs: './results/batched_results*.tsv' // Will automatically number inputs to avoid name collisions

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
    spectra = Channel.fromPath(params.inputspectra + "/**", relative: true)
    
    if(params.searchtool == "gnps"){
        // Perform cartesian product producing all combinations of library, spectra
        inputs = libraries.combine(spectra)

        // For each path, add the path as a string for file naming. Result is [library_file, spectrum_file, spectrum_path_as_str]
        // Must add the prepend manually since relative does not inlcude the glob.
        inputs = inputs.map { it -> [it[0], file(params.inputspectra + '/' + it[1]), it[1].toString().replaceAll("/","_"), it[1]] }

        (search_results) = searchDataGNPS(inputs)

        chunked_results = chunkResults(search_results.buffer(size: params.merge_batch_size, remainder: true))
       
        // Collect all the batched results and merge them at the end
        merged_results = mergeResults(chunked_results.collect())
    }
    else if (params.searchtool == "blink"){
        // Must add the prepend manually since relative does not inlcude the glob.
        spectra = spectra.map { it -> file(params.inputspectra + '/' + it) }
        search_results = searchDataBlink(libraries, spectra)

        formatted_results = formatBlinkResults(search_results)

        merged_results = mergeResults(formatted_results.collect())
    }

    getGNPSAnnotations(merged_results)
}