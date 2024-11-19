import os
import argparse
import pandas as pd
import numpy as np
from file_io import iterate_gnps_lib_mgf, load_qry_file
from _utils import clean_peaks
from cosine import CosineGreedy
from entropy import EntropyGreedy


def main(gnps_lib_mgf, qry_file,
         algorithm='cos',
         pm_tol=0.02, frag_tol=0.05,
         min_score=0.7, min_matched_peak=3,
         rel_int_threshold=0.01, prec_mz_removal_da=1.5, peak_transformation='sqrt',
         ):
    """
    Main function to search GNPS library

    algorithm: str. 'cos' or 'entropy'
    peak_transformation: str. 'sqrt' or 'none'
    """

    if algorithm == 'cos':
        search_eng = CosineGreedy(tolerance=frag_tol, reverse=False)
    elif algorithm == 'rev_cos':
        search_eng = CosineGreedy(tolerance=frag_tol, reverse=True)
    elif algorithm == 'entropy':
        search_eng = EntropyGreedy(tolerance=frag_tol, reverse=False)
    elif algorithm == 'rev_entropy':
        search_eng = EntropyGreedy(tolerance=frag_tol, reverse=True)
    else:
        raise ValueError("Invalid algorithm")

    # load query file as a list of Spectrum objects
    qry_spec_list, all_qry_prec_mz_array = load_qry_file(qry_file)

    all_match_rows = []  # store all matched rows

    # iterate GNPS library
    for spec in iterate_gnps_lib_mgf(gnps_lib_mgf):
        ref_prec_mz = spec['PEPMASS']

        if len(spec['peaks']) == 0:
            continue

        # find all qry spectra within precursor tolerance
        qry_idx_list = np.where(np.abs(all_qry_prec_mz_array - ref_prec_mz) <= pm_tol)[0]
        if len(qry_idx_list) == 0:
            continue

        # clean reference peaks
        ref_peaks = clean_peaks(np.array(spec['peaks']), ref_prec_mz,
                                rel_int_threshold=rel_int_threshold,
                                prec_mz_removal_da=prec_mz_removal_da,
                                peak_transformation=peak_transformation)

        # iterate over all matching query spectra
        for qry_idx in qry_idx_list:
            qry_spec = qry_spec_list[qry_idx]

            if len(qry_spec.peaks) == 0:
                continue

            # check if it has cleaned peaks
            if qry_spec.cleaned_peaks is None:
                qry_spec.cleaned_peaks = clean_peaks(qry_spec.peaks, qry_spec.precursor_mz,
                                                     rel_int_threshold=rel_int_threshold,
                                                     prec_mz_removal_da=prec_mz_removal_da,
                                                     peak_transformation=peak_transformation)

            # calculate similarity score
            # qry comes first, ref comes second (reverse search matters)
            score, n_matches = search_eng.pair(qry_spec, ref_peaks)

            # filter by minimum score and minimum matched peaks
            if score < min_score or n_matches < min_matched_peak:
                continue

            mz_error_ppm = round((qry_spec.precursor_mz - ref_prec_mz) / ref_prec_mz * 1e6, 2)
            mass_diff = round(qry_spec.precursor_mz - ref_prec_mz, 4)

            # store matched rows
            all_match_rows.append({
                '#Scan#': qry_spec.scan,
                'SpectrumFile': qry_spec.file,
                'Annotation': '*..*',
                'OrigAnnotation': '',
                'Protein': '',
                'dbIndex': '',
                'numMods': '',
                'matchOrientation': '',
                'startMass': '',
                'Charge': qry_spec.charge,
                'MQScore': score,
                'p-value': qry_spec.rt,
                'isDecoy': 0,
                'StrictEnvelopeScore': '',
                'UnstrictEvelopeScore': qry_spec.tic,
                'CompoundName': '',
                'Organism': '',
                'FileScanUniqueID': qry_spec.file_scan_id,
                'FDR': -1,
                'LibraryName': '',
                'mzErrorPPM': mz_error_ppm,
                'LibMetaData': '',
                'Smiles': '',
                'Inchi': '',
                'LibSearchSharedPeaks': n_matches,
                'Abundance': '',
                'ParentMassDiff': mass_diff,
                'SpecMZ': qry_spec.precursor_mz,
                'ExactMass': '',
                'LibrarySpectrumID': spec['SPECTRUMID']
            })

    match_df = pd.DataFrame(all_match_rows)

    lib_mgf_basename = os.path.splitext(os.path.basename(gnps_lib_mgf))[0]
    qry_basename = os.path.basename(qry_file).replace("/", "_").replace(".", "_").replace(" ", "_")  # keep the extension, avoid conflicts
    out_name = f"{qry_basename}_{lib_mgf_basename}_matches.tsv"

    out_path = os.path.join('search_results', out_name)

    match_df.to_csv(out_path, sep='\t', index=False)

    return


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description='Search GNPS library')
    argparse.add_argument('--gnps_lib_mgf', type=str, help='GNPS library MGF file path')
    argparse.add_argument('--qry_file', type=str, help='Query file path')
    argparse.add_argument('--algorithm', type=str, default='cos',
                          help='Algorithm: cos, rev_cos, entropy, rev_entropy')
    argparse.add_argument('--pm_tol', type=float, default=0.02, help='Precursor m/z tolerance')
    argparse.add_argument('--frag_tol', type=float, default=0.05, help='Fragment m/z tolerance')
    argparse.add_argument('--min_score', type=float, default=0.7, help='Minimum score')
    argparse.add_argument('--min_matched_peak', type=int, default=3, help='Minimum matched peaks')
    argparse.add_argument('--rel_int_threshold', type=float, default=0.01, help='Relative intensity threshold')
    argparse.add_argument('--prec_mz_removal_da', type=float, default=1.5, help='Precursor m/z removal')
    argparse.add_argument('--peak_transformation', type=str, default='sqrt',
                          help='Peak transformation, sqrt or none')

    args = argparse.parse_args()

    main(args.gnps_lib_mgf, args.qry_file,
         algorithm=args.algorithm,
         pm_tol=args.pm_tol, frag_tol=args.frag_tol,
         min_score=args.min_score, min_matched_peak=args.min_matched_peak,
         rel_int_threshold=args.rel_int_threshold, prec_mz_removal_da=args.prec_mz_removal_da,
         peak_transformation=args.peak_transformation)
