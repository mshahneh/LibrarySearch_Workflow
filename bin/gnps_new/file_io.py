import os
from pyteomics import mzml, mzxml
import numpy as np
from _utils import Spectrum


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


def iterate_mgf(mgf_path, buffer_size=1048576):
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



def process_file(file_path):
    try:
        file_format = os.path.splitext(file_path)[1].lower()
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        output = []
        if file_format == '.mzml':
            reader = mzml.MzML(file_path)
            output = process_mzml(reader, file_name)
        elif file_format == '.mzxml':
            reader = mzxml.MzXML(file_path)
            output = process_mzxml(reader, file_name)
        return output
    except:
        return []


def process_mzml(reader, file_name):
    output = []
    for spectrum in reader:
        if spectrum['ms level'] == 2:  # MS2 scans only
            precursor_mz = round(float(
                spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']), 4)

            mz_array = np.array(spectrum['m/z array'])
            intensity_array = np.array(spectrum['intensity array'])

            peaks = np.column_stack((mz_array, intensity_array))

            # sort by mz
            peaks = peaks[peaks[:, 0].argsort()]

            # normalize intensity, avoiding division by zero
            if peaks.size > 0:
                max_intensity = np.max(peaks[:, 1])
                if max_intensity > 0:
                    peaks[:, 1] = peaks[:, 1] / max_intensity * 999
                else:
                    peaks[:, 1] = 0

            scan_number = spectrum['index'] + 1
            try:
                charge = int(
                    spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state'])
            except:
                charge = 0

            polarity = 1 if 'positive scan' in spectrum else -1
            charge = charge * polarity

            output.append({
                'title': f"{title_name}:{file_name}:scan:{scan_number}",
                'pepmass': precursor_mz,
                'charge': charge,
                'peaks': peaks
            })

    return output


def process_mzxml(reader, file_name):
    output = []
    for spectrum in reader:
        if spectrum['msLevel'] == 2:  # MS2 scans only
            precursor_mz = round(float(spectrum['precursorMz'][0]['precursorMz']), 4)

            mz_array = np.array(spectrum['m/z array'])
            intensity_array = np.array(spectrum['intensity array'])

            peaks = np.column_stack((mz_array, intensity_array))
            # sort by mz
            peaks = peaks[peaks[:, 0].argsort()]
            # normalize intensity, avoiding division by zero
            if peaks.size > 0:
                max_intensity = np.max(peaks[:, 1])
                if max_intensity > 0:
                    peaks[:, 1] = peaks[:, 1] / max_intensity * 999
                else:
                    peaks[:, 1] = 0

            scan_number = spectrum['num']

            if spectrum['polarity'] == '+':
                polarity = 1
            elif spectrum['polarity'] == '-':
                polarity = -1
            else:
                polarity = 0

            try:
                charge = int(spectrum['precursorMz'][0]['precursorCharge'])
            except:
                charge = 0

            charge = charge * polarity

            output.append({
                'title': f"{title_name}:{file_name}:scan:{scan_number}",
                'pepmass': precursor_mz,
                'charge': charge_str,
                'peaks': peaks
            })

    return output