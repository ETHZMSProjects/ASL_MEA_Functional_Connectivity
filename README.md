# 🧠 ASL_MEA_Functional_Connectivity

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

Download the dataset from Dryad:  
👉 https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

Specifically, download tuningTasks.tar.gz and sentences.tar.gz. 

Place the downloaded files in a `data/` directory.

### 4. Generate Granger Causality Results

Run the provided notebooks to generate [whole session](./GC_Whole_Session.ipynb) or [word trial](./GC_Word_Trials.ipynb) results. 
Change the gc_wholefile_dir and dataDir variables to correspond to your local file organization.

### 5. Generate the figures

Run the provided notebooks to generate figures in the project writeup:

- [Figures 3-8](./Channel_Activity_Plots.ipynb)
- [Figure 9](./GC_Word_Trials.ipynb)
- [Figures 10-11](./GC_Whole_Session.ipynb)

---

## 📝 Notes

- All analysis code assumes data is organized in the `data/` subdirectory.
- Scripts were tested with Python 3.12.

---

## 📬 Contact

For questions or contributions, feel free to open an issue. 
