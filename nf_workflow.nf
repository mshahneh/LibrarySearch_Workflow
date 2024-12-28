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

// Filtering structures
params.filtertostructures = "0" // 1 means we filter to only hits with structures

//TODO: Implement This
params.filter_precursor = 1
params.filter_window = 1

//TODO: Implement This
params.analog_search = "0"
params.analog_max_shift = 1999

// Blink Parameters
params.blink_ionization = "positive"
params.blink_minpredict = 0.01

TOOL_FOLDER = "$moduleDir/bin"
MODULES_FOLDER = "$TOOL_FOLDER/NextflowModules"
params.publishDir = "./nf_output"

include {summaryLibrary; searchDataGNPS; searchDataBlink; mergeResults; librarygetGNPSAnnotations; filtertop1Annotations; formatBlinkResults; chunkResults} from "$MODULES_FOLDER/nf_library_search_modules.nf" addParams(publishDir: params.publishDir)

workflow Main{
    take:
    inputlibraries
    inputspectra
    searchtool
    topk
    fragment_tolerance
    pm_tolerance
    library_min_cosine
    library_min_matched_peaks
    merge_batch_size
    filtertostructures
    filter_precursor
    filter_window
    analog_search
    analog_max_shift
    blink_ionization
    blink_minpredict
    publishDir

    main:
    libraries_ch = Channel.fromPath(inputlibraries + "/*.mgf" )
    spectra = Channel.fromPath(inputspectra + "/**", relative: true)

    // Lets create a summary for the library files
    library_summary_ch = summaryLibrary(libraries_ch)

    // Merging all these tsv files from library_summary_ch within nextflow
    library_summary_merged_ch = library_summary_ch.collectFile(name: "${publishDir}/library_summary.tsv", keepHeader: true)
    
    if(searchtool == "gnps"){
        // Perform cartesian product producing all combinations of library, spectra
        inputs = libraries_ch.combine(spectra)

        // For each path, add the path as a string for file naming. Result is [library_file, spectrum_file, spectrum_path_as_str]
        // Must add the prepend manually since relative does not include the glob.
        inputs = inputs.map { it -> [it[0], file(inputspectra + '/' + it[1]), it[1].toString().replaceAll("/","_"), it[1]] }

        (search_results) = searchDataGNPS(inputs, pm_tolerance, fragment_tolerance, topk, library_min_cosine, library_min_matched_peaks, analog_search)

        chunked_results = chunkResults(search_results.buffer(size: merge_batch_size, remainder: true), topk)
       
        // Collect all the batched results and merge them at the end
        merged_results = mergeResults(chunked_results.collect(), topk)
    }
    else if (searchtool == "blink"){
        // Must add the prepend manually since relative does not inlcude the glob.
        spectra = spectra.map { it -> file(inputspectra + '/' + it) }
        search_results = searchDataBlink(libraries_ch, spectra, blink_ionization, blink_minpredict, fragment_tolerance)

        formatted_results = formatBlinkResults(search_results)

        merged_results = mergeResults(formatted_results.collect(), topk)
    }

    annotation_results_ch = librarygetGNPSAnnotations(merged_results, library_summary_merged_ch, topk, filtertostructures)

    // Getting another output that is only the top 1
    filtertop1Annotations(annotation_results_ch)

    emit:
    annotation_results_ch
}

workflow {
    Main(
        params.inputlibraries,
        params.inputspectra,
        params.searchtool,
        params.topk,
        params.fragment_tolerance,
        params.pm_tolerance,
        params.library_min_cosine,
        params.library_min_matched_peaks,
        params.merge_batch_size,
        params.filtertostructures,
        params.filter_precursor,
        params.filter_window,
        params.analog_search,
        params.analog_max_shift,
        params.blink_ionization,
        params.blink_minpredict,
        params.publishDir
    )
}