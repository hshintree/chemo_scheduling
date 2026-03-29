# Real Patient Data

This directory stores digitized PSA time-series for mCRPC patients from
the Zhang *et al.* 2017/2022 adaptive therapy trials.

## File naming convention

```
patient_<id>.csv
```

where `<id>` is a zero-padded integer matching the patient numbering in the
source paper (e.g., `patient_01.csv` through `patient_16.csv`).

## Required CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `day` | int | Days from treatment start (day 0 = first abiraterone dose) |
| `psa` | float | PSA value in ng/mL |
| `on_treatment` | int | 1 if abiraterone given that day, 0 otherwise |

**Example:**

```
day,psa,on_treatment
0,28.4,1
30,21.0,1
60,12.5,1
90,8.2,0
120,10.1,0
150,14.3,0
180,28.6,1
```

Observations do not need to be at regular intervals — the fitting pipeline
interpolates the treatment schedule and integrates the ODE to the exact
observation times.

## How to obtain this data

### Primary source

Zhang J, et al. (2022). *Integrating evolutionary dynamics into treatment of
metastatic castration-resistant prostate cancer.* eLife, 11, e74336.
https://doi.org/10.7554/eLife.74336

Individual patient PSA trajectories are in:
- **Supplementary Figure 1**: Adaptive therapy arm (patients 1–8)
- **Supplementary Figure 2**: Standard-of-care (MTD) arm (patients 9–16)

### Also available in

Brady-Nicholls R, et al. (2020). Prostate-specific antigen dynamics predict
individual responses to intermittent androgen deprivation. *npj Systems Biology
and Applications*, 6, 1.
https://doi.org/10.1038/s41540-020-0133-0
(Supplementary Data S1 contains digitized values for some patients.)

## Digitization with WebPlotDigitizer

WebPlotDigitizer is a free, browser-based tool that extracts numerical data
from plot images. No installation required.

### Step-by-step instructions

1. **Obtain the figure image**
   - Download the eLife 2022 supplementary PDF.
   - Extract individual panel images using `pdfimages -png eLIFE_supp.pdf panels/`
     or take a clean screenshot of each subplot.

2. **Open WebPlotDigitizer**
   - Go to https://automeris.io/WebPlotDigitizer/
   - Click **Load Image** and upload your panel PNG.

3. **Calibrate axes**
   - Select **2D (X-Y) Plot**.
   - Click **Calibrate** and place the four calibration points:
     - Two X-axis reference points (e.g., x=0 and x=500 days)
     - Two Y-axis reference points (e.g., y=0 and y=50 ng/mL)
   - Enter the known values for each point.

4. **Extract the PSA curve**
   - Click **Add Dataset**, then choose **Automatic Extraction**.
   - Select **Color** extraction mode. Use the color picker to click on the PSA
     line itself.
   - Adjust the **Delta** (color tolerance) slider until only the curve pixels
     are highlighted (not background noise).
   - Click **Run Extraction**.

5. **Export**
   - Click **View Data** → **Download CSV**.
   - The file contains columns `x` (day) and `y` (PSA).

6. **Add treatment column**
   - The treatment-on periods are shown as shaded (gray) regions in the figure.
   - Create the `on_treatment` column manually: `1` during shaded intervals,
     `0` during unshaded intervals.
   - Save the final file as `patient_<id>.csv` in this directory.

### Quality check

After digitizing, run:

```bash
python scripts/02_fit_patient_parameters.py --patient data/raw/patient_01.csv
```

The fitting script will plot observed vs. model PSA. A good digitization should
give RMSE < 5% of the baseline PSA value.

## Privacy and licensing

These data are from published peer-reviewed supplementary materials and are
reproduced here solely for non-commercial academic research. Cite the original
papers in any work that uses this data.
