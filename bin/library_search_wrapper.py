#!/usr/bin/python


import sys
import getopt
import os
import json
import argparse
import uuid
from collections import defaultdict

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def search_wrapper(search_param_dict):
    search_files(search_param_dict["spectra_files"], search_param_dict["temp_folder"], search_param_dict["tempresults_folder"], search_param_dict["args"], search_param_dict["params_object"], search_param_dict["library_files"])

def search_files(spectra_files, temp_folder, tempresults_folder, args, params_object, library_files):
    parameter_filename = os.path.join(temp_folder, str(uuid.uuid4()) + ".params")
    output_parameter_file = open(parameter_filename, "w")

    #Search Criteria
    output_parameter_file.write("MIN_MATCHED_PEAKS_SEARCH=%s\n" % (params_object["MIN_MATCHED_PEAKS"][0]))
    output_parameter_file.write("TOP_K_RESULTS=%s\n" % (params_object["TOP_K_RESULTS"][0]))
    output_parameter_file.write("search_peak_tolerance=%s\n" % (params_object["tolerance.Ion_tolerance"][0]))
    output_parameter_file.write("search_parentmass_tolerance=%s\n" % (params_object["tolerance.PM_tolerance"][0]))
    output_parameter_file.write("ANALOG_SEARCH=%s\n" % (params_object["ANALOG_SEARCH"][0]))
    output_parameter_file.write("MAX_SHIFT_MASS=%s\n" % (params_object["MAX_SHIFT_MASS"][0]))

    #Filtering Criteria
    output_parameter_file.write("FILTER_PRECURSOR_WINDOW=%s\n" % (params_object["FILTER_PRECURSOR_WINDOW"][0]))
    output_parameter_file.write("MIN_PEAK_INT=%s\n" % (params_object["MIN_PEAK_INT"][0]))
    output_parameter_file.write("WINDOW_FILTER=%s\n" % (params_object["WINDOW_FILTER"][0]))
    output_parameter_file.write("FILTER_LIBRARY=%s\n" % (params_object["FILTER_LIBRARY"][0]))

    #Scoring Criteria
    output_parameter_file.write("SCORE_THRESHOLD=%s\n" % (params_object["SCORE_THRESHOLD"][0]))

    #Output
    output_parameter_file.write("RESULTS_DIR=%s\n" % (os.path.join(tempresults_folder, str(uuid.uuid4()) + ".tsv")))

    output_parameter_file.write("NODEIDX=%d\n" % (0))
    output_parameter_file.write("NODECOUNT=%d\n" % (1))

    output_parameter_file.write("EXISTING_LIBRARY_MGF=%s\n" % (" ".join(library_files)))

    all_query_spectra_list = []
    for spectrum_file in spectra_files:
        fileName, fileExtension = os.path.splitext(os.path.basename(spectrum_file))
        output_filename = ""

        if spectrum_file.find("mzXML") != -1 or spectrum_file.find("mzxml") != -1 or spectrum_file.find("mzML") != -1:
            output_filename = os.path.join(temp_folder, fileName + ".pklbin")
            cmd = "%s %s %s" % (args.convert_binary, spectrum_file, output_filename)
            print(cmd)
            os.system(cmd)
        else:
            output_filename = os.path.join(temp_folder, os.path.basename(spectrum_file))
            cmd = "cp %s %s" % (spectrum_file, output_filename)
            os.system(cmd)

        #Input
        faked_output_filename = os.path.join(temp_folder, os.path.basename(spectrum_file))
        all_query_spectra_list.append(faked_output_filename)


    output_parameter_file.write("searchspectra=%s\n" % (" ".join(all_query_spectra_list)))
    output_parameter_file.close()

    cmd = "%s ExecSpectralLibrarySearchMolecular %s -ccms_input_spectradir %s -ccms_results_prefix %s -ll 9" % (args.librarysearch_binary, parameter_filename, temp_folder, tempresults_folder)
    print(cmd)
    os.system(cmd)

        #Removing the spectrum
        # try:
        #     os.remove(output_filename)
        # except:
        #     print("Can't remove", output_filename)


def main():
    parser = argparse.ArgumentParser(description='Running library search parallel')
    parser.add_argument('spectrum_file', help='spectrum_file')
    parser.add_argument('library_file', help='library_file')
    parser.add_argument('result_folder', help='output folder for results')
    parser.add_argument('convert_binary', help='conversion binary')
    parser.add_argument('librarysearch_binary', help='librarysearch_binary')
    
    args = parser.parse_args()

    temp_folder = "temp"
    try:
        os.mkdir(temp_folder)
    except:
        print("folder error")

    tempresults_folder = "tempresults"
    try:
        os.mkdir(tempresults_folder)
    except:
        print("folder error")

    print(args)

    # performing the search

    #for param_dict in parameter_list:
    #    search_wrapper(param_dict)
    # print("Parallel to execute", len(parameter_list))
    # ming_parallel_library.run_parallel_job(search_wrapper, parameter_list, 5)


    # """Merging Files and adding full path"""
    # all_result_files = ming_fileio_library.list_files_in_dir(tempresults_folder)
    # full_result_list = []
    # for input_file in all_result_files:
    #     result_list = ming_fileio_library.parse_table_with_headers_object_list(input_file)
    #     full_result_list += result_list

    # for result_object in full_result_list:
    #     mangled_name = os.path.basename(result_object["SpectrumFile"])
    #     full_path = mangled_mapping[mangled_name]
    #     result_object["full_CCMS_path"] = full_path

    # ming_fileio_library.write_list_dict_table_data(full_result_list, os.path.join(args.result_folder, str(uuid.uuid4()) + ".tsv"))









if __name__ == "__main__":
    main()
