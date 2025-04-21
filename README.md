# 🧠 ASL_MEA_Functional_Connectivity
[![DOI](https://zenodo.org/badge/930843447.svg)](https://doi.org/10.5281/zenodo.15254343)
This repository contains code and data analysis tools for investigating functional connectivity in multielectrode array (MEA) recordings during spoken word and sentence production by an ALS patient. The data was originally recorded for use in a brain-computer interface (BCI), as reported by Willett et al in the paper below.

---

## 📄 Related Resources

- 📝 [Project Write-up](./Broca_s_Functional_Connectivity_Writeup_040725.pdf)  
  _Summary of methods, key results, and discussion._

- 📚 [A High Performance Speech Neuroprosthesis](https://www.nature.com/articles/s41586-023-06377-x)  
  _Willett et al., Nature (2023)_

- 📂 [Dataset on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq)  
  _Preprocessed MEA recordings used in this study._

---

## ⚙️ Reproducing the Results

To reproduce the key results and figures from the analysis:

### 1. Clone the repository

```bash
git clone https://github.com/ETHZMSProjects/ASL_MEA_Functional_Connectivity.git
cd ASL_MEA_Functional_Connectivity
```

### 2. Set up the environment

If you're using Anaconda:

```bash
conda env create -f env.yaml
conda activate asl_mea_env
```

Or with `pip` (if you don’t use conda):

```bash
pip install -r requirements.txt
```

### 3. Download the data

The entire dataset is available on Dryad:  
👉 https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

The word and sentence trials are contained in tuningTasks.tar.gz (941MB) and sentences.tar.gz (14.8 GB), respectively. 

Alternatively, you can find the files used in the writeup in the datafiles/raw_data directory on [Google Drive](https://drive.google.com/drive/folders/15R0oX1uf52J9ozRgO0Bv4D00a1oUrliX?usp=sharing).

Place the downloaded files in a `data/` directory.

### 4. Generate Granger Causality Results

Run the provided notebooks to generate [whole session](./GC_Whole_Session.ipynb) or [word trial](./GC_Word_Trials.ipynb) results. 

Change the data_dir variable at the beginning of the notebook to correspond to your local environment.

Running this takes a long time, so you can find the Granger Causality results reported in the writeup on [Google Drive](https://drive.google.com/drive/folders/15R0oX1uf52J9ozRgO0Bv4D00a1oUrliX?usp=sharing) in datafiles.

### 5. Generate the figures

Run the provided notebooks to generate figures in the project writeup:

- [Figures 3-8](./Channel_Activity_Plots.ipynb)
- [Figure 9](./GC_Word_Trials.ipynb)
- [Figures 10-11](./GC_Whole_Session.ipynb)

Remember to change the data_dir variable to correspond with your local environment.

---

## 📝 Notes

- Scripts were tested with Python 3.12.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or email me at aidan.truel@gmail.com. 
