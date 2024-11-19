import os
from pyteomics import mzml, mzxml, mgf
import numpy as np
from ._utils import Spectrum


def read_mgf_spectrum(file_obj):
    """Read a single spectrum block from an open MGF file.

    Args:
        file_obj: An opened file object positioned at the start of a spectrum

    Returns:
        dict: Spectrum information containing all metadata and peaks,
        or None if end of file is reached
    """
    # Initialize spectrum with common metadata fields
    spectrum = {
        'PEPMASS': 0.0,
        'CHARGE': '',
        'MSLEVEL': '',
        'SOURCE_INSTRUMENT': '',
        'FILENAME': '',
        'SEQ': '',
        'IONMODE': '',
        'ORGANISM': '',
        'NAME': '',
        'PI': '',
        'DATACOLLECTOR': '',
        'SMILES': '',
        'INCHI': '',
        'INCHIAUX': '',
        'PUBMED': '',
        'SUBMITUSER': '',
        'LIBRARYQUALITY': '',
        'SPECTRUMID': '',
        'SCANS': '',
        'peaks': []
    }

    # Skip any empty lines before BEGIN IONS
    for line in file_obj:
        if line.strip() == 'BEGIN IONS':
            break
    else:  # EOF reached
        return None

    # Read spectrum metadata and peaks
    for line in file_obj:
        line = line.strip()

        if not line:  # Skip empty lines
            continue

        if line == 'END IONS':
            if spectrum['peaks']:  # Only return if we have peaks
                return spectrum
            break

        # Handle peak data
        if line and not line.startswith(('BEGIN', 'END')):
            try:
                # First try to parse as metadata
                key, value = line.split('=', 1)

                # Handle specific numeric fields
                if key == 'PEPMASS':
                    spectrum[key] = float(value.split()[0])  # Handle additional intensity value
                elif key in spectrum:  # Store other known metadata fields as strings
                    spectrum[key] = value
            except ValueError:
                # If no '=' found, treat as peak data
                try:
                    mz, intensity = line.split()
                    spectrum['peaks'].append((float(mz), float(intensity)))
                except ValueError:
                    # Skip lines that can't be parsed
                    continue

    return None


def iterate_gnps_lib_mgf(mgf_path, buffer_size=1048576):
    """Iterate through spectra in an MGF file efficiently using buffered reading.

    Args:
        mgf_path: Path to the MGF file
        buffer_size: Read buffer size in bytes

    Yields:
        dict: Spectrum information containing all metadata and peaks
    """
    with open(mgf_path, 'r', buffering=buffer_size) as f:
        while True:
            spectrum = read_mgf_spectrum(f)
            if spectrum is None:
                break
            yield spectrum


def load_qry_file(file_path):
    try:
        file_format = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if file_format == '.mzml':
            reader = mzml.MzML(file_path)
            output, all_prec_mz_array = process_mzml(reader, file_name)
        elif file_format == '.mzxml':
            reader = mzxml.MzXML(file_path)
            output, all_prec_mz_array = process_mzxml(reader, file_name)
        elif file_format == '.mgf':
            reader = mgf.MGF(file_path)
            output, all_prec_mz_array = process_mgf(reader, file_name)
        else:
            print(f"Unsupported file format: {file_format}")
            return [], np.array([])
    except:
        return [], np.array([])

    return output, all_prec_mz_array


def process_mzml(reader, file_name):
    all_prec_mz_array = []
    output = []
    for spectrum in reader:
        if spectrum['ms level'] == 2:  # MS2 scans only
            precursor_mz = float(spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])

            mz_array = np.array(spectrum['m/z array'])
            intensity_array = np.array(spectrum['intensity array'])

            # remove empty peaks
            if mz_array.size == 0:
                continue

            peaks = np.column_stack((mz_array, intensity_array))
            tic = round(np.sum(intensity_array))

            scan_number = spectrum['index'] + 1

            try:
                rt = float(spectrum['scanList']['scan'][0]['scan start time'])
            except:
                rt = 0

            if 'positive scan' in spectrum:
                polarity = 1
            elif 'negative scan' in spectrum:
                polarity = -1
            else:
                polarity = 0

            try:
                charge = int(
                    spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state'])

                charge = charge * polarity
            except:
                charge = polarity

            output.append(
                Spectrum(file=file_name,
                         scan=scan_number,
                         precursor_mz=precursor_mz,
                         rt=rt,
                         charge=charge,
                         tic=tic,
                         peaks=peaks))
            all_prec_mz_array.append(precursor_mz)

    return output, np.array(all_prec_mz_array)


def process_mzxml(reader, file_name):
    all_prec_mz_array = []
    output = []
    for spectrum in reader:
        if spectrum['msLevel'] == 2:  # MS2 scans only
            precursor_mz = float(spectrum['precursorMz'][0]['precursorMz'])

            mz_array = np.array(spectrum['m/z array'])
            intensity_array = np.array(spectrum['intensity array'])

            # remove empty peaks
            if mz_array.size == 0:
                continue

            peaks = np.column_stack((mz_array, intensity_array))
            tic = round(np.sum(intensity_array))

            scan_number = spectrum['num']

            try:
                rt = float(spectrum['retentionTime'])
            except:
                rt = 0

            try:
                if spectrum['polarity'] == '+':
                    polarity = 1
                elif spectrum['polarity'] == '-':
                    polarity = -1
                else:
                    polarity = 0
            except:
                polarity = 0

            try:
                charge = int(spectrum['precursorMz'][0]['precursorCharge'])
                charge = charge * polarity
            except:
                charge = polarity

            output.append(
                Spectrum(file=file_name,
                         scan=scan_number,
                         precursor_mz=precursor_mz,
                         rt=rt,
                         charge=charge,
                         tic=tic,
                         peaks=peaks))
            all_prec_mz_array.append(precursor_mz)

    return output, np.array(all_prec_mz_array)


def process_mgf(reader, file_name):
    all_prec_mz_array = []
    output = []
    scan_idx = 0  # 1-based index actually
    for spectrum in reader:
        scan_idx += 1

        d = spectrum['params']

        try:
            if 'precursor_mz' not in d:
                precursor_mz = float(d['pepmass'][0])
            else:
                precursor_mz = float(d['precursor_mz'])
        except:
            continue

        mz_array = np.array(spectrum['m/z array'])
        intensity_array = np.array(spectrum['intensity array'])

        if mz_array.size == 0:
            continue

        peaks = np.column_stack((mz_array, intensity_array))
        tic = round(np.sum(intensity_array))

        if 'rtinseconds' in d:
            rt = float(d['rtinseconds'])
        elif 'rt' in d:
            rt = float(d['rt'])
        else:
            rt = 0

        try:
            charge = int(d['charge'][0])
        except:
            charge = 0

        output.append(
            Spectrum(file=file_name,
                     scan=scan_idx,
                     precursor_mz=precursor_mz,
                     rt=rt,
                     charge=charge,
                     tic=tic,
                     peaks=peaks))
        all_prec_mz_array.append(precursor_mz)

    return output, np.array(all_prec_mz_array)


if __name__ == "__main__":
    # load_qry_file('/Users/shipei/Documents/test_data/mzXML/1002.D_GE7_01_4308.mzXML')
    # load_qry_file('/Users/shipei/Documents/test_data/mgf/CASMI.mgf')

    for spec in iterate_gnps_lib_mgf('/Users/shipei/Documents/test_data/mgf/CASMI.mgf'):
        print(spec)
        break
