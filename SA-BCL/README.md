# SA-BCL

## Python library dependencies:
* `Python` -v: `3.8`
* `matplotlib` -v: `3.8.2`
* `numpy` -v: `1.26.3`
* `imbalanced-learn` -v: `0.6.0`
* `scikit-learn` -v: `1.4.0`
* `cleanlab` -v: `0.1.1` 

## Dataset:
Reference paper: Dylan Callaghan and Bernd Fischer "Improving Spectrum-Based Localization of Multiple Faults by
Iterative Test Suite Reduction." Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis. ISSTA, 2023. (link: https://dl.acm.org/doi/pdf/10.1145/3597926.3598148) <br>
The `Data` directory consists of two subdirectories: `TCM` and `Defects4j`. Its structure is as follows:
```
Data
│
└─TCM
│ │
│ └─daikon
│ │ │
│ │ └─1-fault
│ │ │
│ │ └─2-fault
│ │ │
│ │ └─...
│ └─...
│
└─Defects4j
  │
  └─Chart
  │ │
  │ └─Chart-3-4
  │ │
  │ └─Chart-8-9
  │ │
  │ └─...
  └─...
```
Each `n-fault` directory in each project for the TCM evaluation contains the
necessary input files, and each `<Project>-i-j-...` directory in the Defects4J
evaluation contains its corresponding input files.

## Result generation:
If you want to reproduce our experiments, follow these steps to configure your environment 
and run the necessary scripts:
### Configure your environment<br>
Open the .bashrc file:
```
vim ~/.bashrc
```
Add the following lines:
```
export FLITSR_HOME="path/to/replication/package/scripts"
export PATH="${PATH}:$FLITSR_HOME"
```
Replace path/to/replication/package/scripts with the actual path to the scripts directory 
of each RQ in the replication package. Apply the changes by running:
```
source ~/.bashrc
```

### Run the results for the datasets
Navigate to the `Data` directory, and select either the `TCM` or `Defects4j` subdirectory.<br>
To run the results for the TCM dataset, navigate to the `TCM` directory and execute:
```
cd path/to/replication/package/Data/TCM
run_all tcm
```
To run the results for the Defects4J dataset, navigate to the `Defects4j` and execute:
```
cd path/to/replication/package/Data/Defects4j
run_all
```
The results will be saved in the corresponding subdirectories inside 
the `TCM` direcory or `Defects4j` you are working in.

## Results:
The results for all experiments are stored in `results` files, 
both in the current working directory 
and in the subdirectories for individual projects within each dataset. 
This organization ensures easy access to both overall and project-specific results.

### For the Defects4J dataset
- The aggregated results for the entire dataset are saved in the results file located in the `Defects4j` directory.<br>
- Results for individual projects, such as the Chart project, are stored in the results file within the corresponding project subdirectory (e.g., `Defects4j/Chart/results`).
### For the TCM dataset
- Results for different fault numbers are saved in files named `results-<n>-fault` within the `TCM` directory, where `<n>` represents the number of faults.<br>
- Results for various fault numbers in the top level directory, averaged over all projects.


