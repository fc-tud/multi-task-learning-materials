# Multi-Task Learning in Materials Design

The code in this repository accompanies the paper _“Leveraging Multi-Task Learning Regressor Chains for Small and Sparse Tabular Data in Materials Design“_.

## Getting Started 

A basic Linux machine with an installation of Anaconda is able to run the code.
With the command:
```
python main.py
```
the training starts with the configurations set in the `config.py` file. The used model is set in the `config.py` file and starts for all datasets in the specified folder `DATA_FOLDER`.
Ensure that the appropriate conda environment has been activated beforehand. The different conda environments can be found in the `envs` folder.

We recommend starting the script from a `tmux` session. 

## Console Output
### Model Training
```
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
Dataset name: 'dataset-name'
Sparse mode: 'sparse-mode', Data available: 'sparse-level'
train_size for outer loop = 'train-size'
Workdir: 'workdir'
---------- SPLIT 1 ---------- (outer split)
Dataset-Size
With Nan in Label 'Task': 
WithOUT Nan in Label 'Task': 
*Starting Time*
Training in 'training-mode' of 'task' over 'training-time' min started
*End Time*

Results from this outer split

---------- SPLIT 2 ---------- (outer split)
Dataset-Size
With Nan in Label 'Task': 
WithOUT Nan in Label 'Task': 
*Starting Time*
Training in 'training-mode' of 'task' over 'training-time' min started
*End Time*

Results from this outer split

[...all outer splits...]


vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

Next Dataset

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```

## Saved Results 

### Model Training
All results are saved in the directory `workdir` in the following folder: `franework\dataset-name_sparse-mode_sparse-level_start-time`.  
In this folder is saved the file `regression_summary.csv`, in which all performance metrics for each fold are stored.  
In the folders `split_'x'` the outputs of the ML frameworks are stored, which are framework-specific.


### Plot Results
The code for visualizing the results and reproducing the figures from the paper can be found in the following script `results/plot_results.ipynb`.


## Datasets and License

Only a subset of the datasets used in our publication is included in this repository to preserve copyright.
The subset is sufficient as minimal working example and stored in the folder `data/storage`.
It can be used to test the code and it displays the required data structure for the framework.
The subset contains a slightly formatted version of `Guo-2019`, `Yin-2021` and `Xiong-2014`: 

* `Guo-2019`
    - Published as attachment to "Guo, S.; Yu, J; Liu, X.; Wang, C.; Jiang Q.: "A predicting model for properties of steel using the industrial big data based on machine learning." Computational Materials Science 160:95-104 (2019)."
    - Available on Mendeley Data under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/): https://data.mendeley.com/datasets/msf6jzm52g/1
	- The dataset here represents a reduction of the original dataset to 1000 data points. The index of the data points in the original dataset is given in `guo_reduced_1000_index.txt`
* `Yin-2021`
    - Published as attachment to "Yin, B. B.; Liew, K. M.: "Machine learning and materials informatics approaches for evaluating the interfacial properties of fiber-reinforced composites." Composite Structures 273:114328 (2021)."
    - Available on github without further specification of a license: https://github.com/Binbin202/ML-Data
* `Xiong-2014`
    - Published in the Paper "Xiong, J.; Zhang, G.; Hu, J.; Wu, L.: "Bead geometry prediction for robotic GMAW-based rapid manufacturing through a neural network and a second-order regression analysis." J Intell Manuf 25:157–163  (2014)."

The remaining datasets used in our publication can be obtained from the respective owners upon reasonable request or by accessing the original publications. 


