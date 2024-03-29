import pandas as pd
import pymzml
import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm
import pyteomics
from pyteomics import mzml, auxiliary, mgf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import json
from rdkit import Chem
import argparse

def _read_mgf(mgf_path: str) -> pd.DataFrame:
    msms_df = []
    with mgf.MGF(mgf_path) as reader:
        for spectrum in reader:
            d = spectrum['params']
            d['spectrum'] = np.array([spectrum['m/z array'],
                                      spectrum['intensity array']])
            if 'precursor_mz' not in d:
                d['precursor_mz'] = d['pepmass'][0]
            else:
                d['precursor_mz'] = float(d['precursor_mz'])
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df

def main(annotations, msms_df):
    contains_educt = msms_df[~msms_df['online_reactivity'].isna()]

    annotations['reactivity'] = 0

    for index, row in contains_educt.iterrows():
        reaction_smarts = []
        sample = json.loads(row['online_reactivity'])
        for item in sample:
            reaction_smarts.append(Chem.MolFromSmarts(item['educt_smarts']))

        scanNumber = int(row['scans'])
        annotations_row = annotations[(annotations['#Scan#'] == scanNumber) & (~annotations['Smiles'].isna())]

        for i, match_row in annotations_row.iterrows():
            try:
                Mol = Chem.MolFromSmiles(match_row['Smiles'])
                for substruct in reaction_smarts:
                    if Mol.HasSubstructMatch(substruct):
                        annotations.loc[i, 'reactivity'] = annotations.loc[i, 'reactivity'] + 1
            except:
                pass

    annotations.to_csv(args.gnps_annotations, sep="\t")

if __name__ == "__main__":
    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Merging Library results Files')
    parser.add_argument('gnps_annotations', help='gnps_annotations')
    parser.add_argument('spectrum_file', help='spectrum_file')

    args = parser.parse_args()

    annotations = pd.read_csv(args.gnps_annotations, sep="\t")
    # if the spectrum_file is not MGF, return
    if not args.spectrum_file.endswith(".mgf"):
        annotations.to_csv(args.gnps_annotations)
    else:
        # if the spectrum_file is MGF, read the MGF file
        msms_df = _read_mgf(args.spectrum_file)
        main(annotations, msms_df)