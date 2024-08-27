#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.inputlibraries = "data/libraries"
params.inputspectra = "data/spectra"

// Parameters
params.searchtool = "gnps" // blink, gnps, gnps_new

params.topk = 1

params.fragment_tolerance = 0.5
params.pm_tolerance = 2.0

params.library_min_similarity = 0.7
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

// GNPS_New Parameters
params.search_algorithm = "cos"
params.peak_transformation = 'sqrt'
params.unmatched_penalty_factor = 0.6

// Blink Parameters
params.blink_ionization = "positive"
params.blink_minpredict = 0.01

TOOL_FOLDER = "$moduleDir/bin"
MODULES_FOLDER = "$TOOL_FOLDER/NextflowModules"
params.publishDir = "./nf_output"

include {summaryLibrary; searchDataGNPS; searchDataGNPSNew; searchDataBlink; 
 mergeResults; librarygetGNPSAnnotations; filtertop1Annotations;
  formatBlinkResults; chunkResults} from "$MODULES_FOLDER/nf_library_search_modules.nf" addParams(publishDir: params.publishDir)

workflow Main{
    take:
    input_map

    main:
    libraries_ch = Channel.fromPath(inputlibraries + "/*.mgf" )
    spectra = Channel.fromPath(inputspectra + "/**", relative: true)

    // Lets create a summary for the library files
    library_summary_ch = summaryLibrary(libraries_ch)

    // Merging all these tsv files from library_summary_ch within nextflow
    library_summary_merged_ch = library_summary_ch.collectFile(name: "library_summary.tsv", keepHeader: true)
    
    if(input_map.searchtool == "gnps"){
        // Perform cartesian product producing all combinations of library, spectra
        inputs = libraries_ch.combine(spectra)

        // For each path, add the path as a string for file naming. Result is [library_file, spectrum_file, spectrum_path_as_str]
        // Must add the prepend manually since relative does not include the glob.
        inputs = inputs.map { it -> [it[0], file(input_map.inputspectra + '/' + it[1]), it[1].toString().replaceAll("/","_"), it[1]] }

        (search_results) = searchDataGNPS(inputs, input_map.pm_tolerance, input_map.fragment_tolerance, input_map.topk,
         input_map.library_min_cosine, input_map.library_min_matched_peaks, input_map.analog_search)

        chunked_results = chunkResults(search_results.buffer(size: input_map.merge_batch_size, remainder: true), input_map.topk)
       
        // Collect all the batched results and merge them at the end
        merged_results = mergeResults(chunked_results.collect(), input_map.topk)
    }
    else if (input_map.searchtool == "blink"){
        // Must add the prepend manually since relative does not inlcude the glob.
        spectra = spectra.map { it -> file(input_map.inputspectra + '/' + it) }
        search_results = searchDataBlink(libraries_ch, spectra, input_map.blink_ionization, input_map.blink_minpredict, input_map.fragment_tolerance)

        formatted_results = formatBlinkResults(search_results)

        merged_results = mergeResults(formatted_results.collect(), input_map.topk)
    }
    else if (input_map.searchtool == "gnps_new"){
        spectra_abs = Channel.fromPath(input_map.inputspectra + "/**", relative: false)

        // Perform cartesian product producing all combinations of library, spectra
        inputs = libraries_ch.combine(spectra_abs)

        search_results = searchDataGNPSNew(inputs)

        merged_results = mergeResults(search_results.collect())
    }

    annotation_results_ch = librarygetGNPSAnnotations(merged_results, library_summary_merged_ch, topk, filtertostructures)

    // Getting another output that is only the top 1
    filtertop1Annotations(annotation_results_ch)

    emit:
    annotation_results_ch
}

workflow {
    input_map = [
        inputlibraries: params.inputlibraries,
        inputspectra: params.inputspectra,
        searchtool: params.searchtool,
        topk: params.topk,
        fragment_tolerance: params.fragment_tolerance,
        pm_tolerance: params.pm_tolerance,
        library_min_cosine: params.library_min_cosine,
        library_min_matched_peaks: params.library_min_matched_peaks,
        merge_batch_size: params.merge_batch_size,
        filtertostructures: params.filtertostructures,
        filter_precursor: params.filter_precursor,
        filter_window: params.filter_window,
        analog_search: params.analog_search,
        analog_max_shift: params.analog_max_shift,
        blink_ionization: params.blink_ionization,
        blink_minpredict: params.blink_minpredict,
        publishDir: params.publishDir
    ]
    
    Main(input_map)
}