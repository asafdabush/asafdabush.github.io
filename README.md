# Bio Medical EEG Analysis for Parkinson's Diagnosis

This repository contains code and documentation for replicating the study
“Diagnosis of Parkinson's disease using higher order statistical analysis of alpha and beta rhythms”.
The goal of this project is to preprocess electroencephalogram (EEG) data,
extract non‑linear and higher‑order statistical features from ICA components,
and perform classification of Parkinson's disease (PD) versus healthy controls (HC).

## Project Overview

1. **Preprocessing:** The script `preprocessing_paper_faithful.m` implements a faithful
   recreation of the preprocessing pipeline described in the paper. Raw EEG data
   is resampled to 500 Hz, band‑pass filtered (0.1–100 Hz), and cleaned of eye (EOG) and
   muscle (EMG) artifacts using Adaptive Artifact Removal (AAR). FastICA is then applied
   to obtain independent components (ICs). The IC signals are segmented into
   2‑second non‑overlapping windows and saved as EEGLAB `.set` files for three band types:
   the full spectrum, the alpha band (8–12 Hz), and the beta band (13–30 Hz).

2. **Feature Extraction:** The function
   `PD_HOS_Full_Extract_PaperFaithful_IC_NO_APPROX.m` loads the segmented IC datasets
   and computes a comprehensive set of features for each component in each epoch. The features include:

   - **Nonlinear dynamics:** Lyapunov exponents (LLE), Hurst exponent (HE), approximate entropy (ApEn),
     and correlation dimension (CD) computed per IC segment.
   - **Higher‑order moments:** Third through eighth central moments (Mu3–Mu8) capturing deviations from Gaussianity.
   - **Bispectral/Bicepstral statistics:** Measures (H1–H5, BH1–BH5) and entropies (BE1, BE2, BEp)
     computed from the bispectrum and bicepstrum of each IC segment.

   The extracted features for each segment are saved into a CSV file with a fixed number of
   independent components (44 per segment) to ensure consistent columns across subjects and bands.

3. **Classification:** The script `segment_level_kfold.m` performs segment‑level classification
   using two classifiers—Bagged Trees (random forest–like ensemble) and Support Vector Machines (SVM) with a
   radial basis function (RBF) kernel. Ten‑fold cross‑validation is applied on all segments to
   assess accuracy, sensitivity, specificity, and F1‑score. Receiver Operating Characteristic (ROC) curves
   are plotted and saved for each band. A summary table of performance metrics is also produced.

## Usage

1. **Preprocessing:**

   ```matlab
   % Run from MATLAB with EEGLAB installed
   preprocessing_paper_faithful;
   ```

   This will create a `Preprocessed_Article_ICA_PaperFaithful` directory within the
   base folder containing segmented `.set` files for AllBands, Alpha, and Beta.

2. **Feature Extraction:**

   ```matlab
   PD_HOS_Full_Extract_PaperFaithful_IC_NO_APPROX;
   ```

   The script writes the extracted features into `Final_Features_PaperFaithful_IC.csv`. Adjust
   `MIN_ICS` and other parameters as needed.

3. **Classification:**

   ```matlab
   segment_level_kfold;
   ```

   Ensure the `csvFile` path inside the script points to the feature CSV. The script will
   save ROC curves and a summary of classification performance in the specified output directory.

## Documentation

The file [`Bio_medical_summery.pdf`](Bio_medical_summery.pdf) (in Hebrew) provides a
concise summary of the methodology, datasets, preprocessing steps, feature definitions,
and results described in the original research. You can refer to it for
a detailed explanation of the approach.

## Notes

* The scripts in this repository rely on EEGLAB and MATLAB toolboxes such as
  the Parallel Computing Toolbox. Make sure these dependencies are installed.
* The provided scripts assume a specific folder structure and naming convention for
  subjects and bands. You may need to adapt paths and file names to your data.
* This project is intended as an educational and reproducible implementation of the
  referenced paper. It does not include the raw EEG datasets.
