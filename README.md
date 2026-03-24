# Code for Wasserstein et al. (2026)

This repository contains the code and data used to produce the figures and analyses in:

> Wasserstein et al. (2026). *Orographic Influences on Precipitation in a Continental Mountain Environment as Observed by Mountain and Valley Profiling Radars*. *Monthly Weather Review*.

## Overview

Analysis of Micro Rain Radar (MRR) and PARSIVEL disdrometer observations from two mountain sites in the Wasatch Range, Utah, during the 2022–23 and 2023–24 cool seasons (November–April). Precipitation characteristics are stratified by synoptic storm type:

| Label | Description |
|-------|-------------|
| **S/SWIVT** | South or southwest integrated vapor transport events |
| **FR** | Frontal events |
| **PF** | Northwest post-cold-frontal events |
| **ALL** | All available 12-h periods |

**Sites:**
- Alta Atwater (Alta, UT)
- Highland High School (Highland, UT)

**Authors:** Michael Wasserstein, Ashley Evans, Jim Steenburgh, Dave Kingsmill, Peter Veals

**Data:** The MRR and PARSIVEL datasets used for this code can be obtained from the University of Utah Research Data Repository:

1. Alta Micro Rain Radar - https://hive.utah.edu/concern/datasets/nk322d41g
2. Highland Micro Rain Radar - https://hive.utah.edu/concern/datasets/h989r329k
3. Alta PARSIVEL - https://hive.utah.edu/concern/datasets/rb68xb93w
4. Highland PARSIVEL - https://hive.utah.edu/concern/datasets/9306sz402

---

## Repository Structure

```
.
├── Data/                          # Input data, must be processed by user
│   ├── MRR/                       # Processed MRR NetCDF files by site and event type
│   ├── PARSIVEL/                  # Processed PARSIVEL .npy arrays by event type
│   └── *.csv                      # Alta Collins precipitation event lists
├── Fig/                           # Output figures
├── MRR_functions.py               # Core MRR data ingest and plotting functions
├── Parsivel_inputs_hgh.py         # PARSIVEL configuration for Highland site
├── Generate_MRR_Datasets.py       # Generate MRR NetCDF datasets by event type
├── Generate_PARSIVEL_Datasets.py  # Generate PARSIVEL matrices by event type
├── Plot_CFADs_classification.ipynb
├── Plot_Composites_v2_Sampled_Events.ipynb
├── Plot_Echo_Top_Height_Histogram.ipynb
├── Plot_Individual_Event_MRR.ipynb
├── Plot_Individual_Event_MRR_Vertical_Reflectivity_Gradient.ipynb
├── Plot_PARSIVEL_Matrices.ipynb
├── Plot_Profiles_5dBZ.ipynb
├── Plot_Seasonal_Accumulation.ipynb
└── Plot_Vertical_Reflectivity_Gradient.ipynb
```

---

## Data Generation Scripts

Before running the plotting notebooks, the processed datasets must be generated, for each synoptic classification:

```bash
# Generate MRR datasets (options: ALL, FR, SIVT, PF)
python Generate_MRR_Datasets.py --period FR

# Generate PARSIVEL datasets (options: ALL, FR, SIVT, PF)
python Generate_PARSIVEL_Datasets.py --period FR
```

These scripts read raw observational data from the data archive and write processed files to `Data/MRR/` and `Data/PARSIVEL/`.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Plot_CFADs_classification.ipynb` | CFADs of MRR reflectivity, Doppler velocity, and spectral width by storm type at Alta and Highland |
| `Plot_Composites_v2_Sampled_Events.ipynb` | Synoptic composite plots (500-hPa heights, 700-hPa temp/winds, IVT) for each storm type at the midpoint of the storm |
| `Plot_Echo_Top_Height_Histogram.ipynb` | Histograms of MRR echo top height (MSL and AGL) by storm type |
| `Plot_Individual_Event_MRR.ipynb` | Time-height MRR cross-sections and CFADs for three representative case studies |
| `Plot_Individual_Event_MRR_Vertical_Reflectivity_Gradient.ipynb` | Vertical Reflectivity histograms for three representative case study events |
| `Plot_PARSIVEL_Matrices.ipynb` | PARSIVEL fall speed vs. diameter matrices by storm type |
| `Plot_Profiles_5dBZ.ipynb` | Vertical profiles of Ze ≥ 5 dBZ frequency by storm type |
| `Plot_Seasonal_Accumulation.ipynb` | Seasonal snow and liquid precipitation equivalent accumulation at Collins Alta by storm type |
| `Plot_Vertical_Reflectivity_Gradient.ipynb` | Histograms of the vertical reflectivity gradient (VRG; dBZ km⁻¹) by storm type |

---

## Dependencies

- Python 3.x
- `numpy`
- `pandas`
- `xarray`
- `matplotlib`
- `scipy`
- `metpy`
- `cartopy`

---
