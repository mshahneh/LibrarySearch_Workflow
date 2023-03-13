mkdir libraries && cd libraries
wget https://external.gnps2.org/gnpslibrary/GNPS-LIBRARY.mgf
wget https://external.gnps2.org/gnpslibrary/GNPS-SELLECKCHEM-FDA-PART1.mgf
cd ../ && mkdir spectra && cd spectra
wget --output-document=isa_9.mzML "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000084030/ccms_peak/isa_9.mzML&forceDownload=true"