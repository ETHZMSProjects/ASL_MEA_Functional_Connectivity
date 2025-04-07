# 🧠 ASL_MEA_Functional_Connectivity

This repository contains code and data analysis tools for investigating functional connectivity in multielectrode array (MEA) recordings during spoken word and sentence production by an ALS patient.

---

## 📄 Related Resources

- 📝 [Project Write-up](./Broca_s_Functional_Connectivity_Writeup_040725.pdf)  
  _Summary of methods, key results, and discussion._

- 📚 [SpeechBCI Paper](https://www.nature.com/articles/s41586-023-06377-x)  
  _Willett et al., Nature (2023)_

- 📂 [Dataset on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq)  
  _Raw and preprocessed MEA recordings used in this study._

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
conda env create -f environment.yml
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

Place the downloaded files in the `data/` directory.

### 4. Generate Granger Causality Results

```bash
python analysis/run_connectivity_analysis.py
```

### 5. Generate the figures

```bash
python figures/plot_gc_summary.py
```

---

## 📝 Notes

- All analysis code assumes data is organized in the `data/` subdirectory.
- Figures and intermediate outputs will be saved in the `figures/` and `results/` folders.
- Scripts were tested with Python 3.12.

---

## 📬 Contact

For questions or contributions, feel free to open an issue. 
