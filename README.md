# GA DSI Capstone: Walking Gait Analysis via Machine Learning

# Table of contents
1. [Project Description](#1-project-description) <br>
    1.1. [Problem Statement](#11-problem-statement) <br>
    1.2. [Background](#12-background) <br>
    1.3. [Data](#13-data)  <br>
    1.4. [Overview](#14-overview)  <br>
2. [Results and learnings](#2-results-and-learnings) <br>
    2.1. [Training and evaluation results](#21-training-and-evaluation-results) <br>
    2.2. [Model Selection and Conclusion](#22-model-selection-and-conclusion) <br>
3. [References](#3-references) <br>
<br>

# 1. Project Description
[[back to the top]](#table-of-contents) <br>

## 1.1. Problem Statement ##
[[back to the top]](#table-of-contents) <br>
Using machine learning model on walking data from GaitRec database, this project aims to infer key features that relates with different walking issues. As an extension to domain knowledge, we can use additional features indentified in this project to help suggest affordable wearable sensors that target these key features that can help clinicians monitor the outcomes of treatment. <br>

The target audience will be general public, carers, physio and occupational therapists. <br>

## 1.2. Background ##
[[back to the top]](#table-of-contents) <br>

In addition to Singapore’s ageing population, there are many factors affecting walking gait, balance, mobility and quality of life. <br>

Common issues affecting walking are: <br>
* Osteoarthritis
* Parkinson's disease
* Stroke
* Muscular atropy
* Hip fractures 
* Musculoskeletal injuries
* Loss of balance

<br>

## 1.3. Data ##    
[[back to the top]](#table-of-contents) <br>

Publicly available online database: [1]
* https://springernature.figshare.com/collections/GaitRec_A_large-scale_ground_reaction_force_dataset_of_healthy_and_impaired_gait/4788012 

<br>
    
GAITREC is a large dataset containing bi-lateral GRF walking trials of over 2000 patients with labelled gait disorders, as well as over 200 healthy controls.
It includes joint replacement, fractures, ligament ruptures, and related disorders at the hip, knee, ankle or calcaneus during entire stay in rehabilitation center. [1]

<br>

* Each of the 2000+ participants is labelled by experts: 
|Classification| Abbrevation|
|--|--|
|Healthy Controls| HC|
|Gait Disorders|GD|
|Gait Disorders - Knee|K|
|Gait Disorders - Hip|H|
|Gait Disorders - Ankle|A|
|Gait Disorders - Calcaneus|C|


**Data Dictionary** 
|Dataset| Description| Data shape| Data type|
|--|--|--|--|
|Vertical GRF|Vertical Ground Reaction Force taken for participant during a session|1xn|double|
|Anterior-Posterior GRF| Breaking and propulsive shear force taken for participant during a session|1xn|double|
|Medio-lateral GRF| Side-to-side shear force taken for participant during a session|1xn|double|
|Anterior-Posterior COP| Center of Pressure coordinate in walking direction taken for participant during a session|1xn|double|
|Medio-lateral COP| Center of Pressure coordinate in side-to-side direction taken for participant during a session|1xn|double|
|Labels|Gait labelling for each of the participant done by experts |NA|NA|

Notes:<br>
* GRF: Ground reaction force in newtons (N)
* COP: Center of Pressure coordinate (cm)

**Engineered Features** 
|Feature| Description| Data shape| Data type|
|--|--|--|--|
|SUBJECT_ID|Participant Identifier|1|int64|
|SESSION_ID|Session Identifier|1|int64|
|TRIAL_ID|Trial number|1|int64|
|c_ml_min_r|Right side-to-side minimum Center of Pressure|1|float64|
|c_ml_max_r|Right side-to-side maximum Center of Pressure|1|float64|
|c_ml_min_l|Left side-to-side maximum Center of Pressure|1|float64|
|c_ml_max_l|Left side-to-side minimum Center of Pressure|1|float64|
|c_ap_slope_r|Right walk-direction change in Center of Pressure|1|float64|
|c_ap_slope_l|Left walk-direction change in Center of Pressure|1|float64|
|g_ml_max1_r|Right side-to-side landing in Ground reaction force|1|float64|
|g_ml_min_r|Right side-to-side transfer Ground reaction force|1|float64|
|g_ml_max2_r|Right side-to-side push-off Ground reaction force|1|float64|
|g_ml_max1_l|Left side-to-side landing in Ground reaction force|1|float64|
|g_ml_min_l|Left side-to-side transfer Ground reaction force|1|float64|
|g_ml_max2_l|Left side-to-side push-off Ground reaction force|1|float64|
|g_ap_min_r|Right walking-direction landing Ground reaction force|1|float64|
|g_ap_max_r|Right walking-direction push-off Ground reaction force|1|float64|
|g_ap_min_l|Left walking-direction landing Ground reaction force|1|float64|
|g_ap_max_l|Left walking-direction push-off Ground reaction force|1|float64|
|g_v_max1_r|Right vertical landing Ground reaction force|1|float64|
|g_v_min_r|Right vertical transfer Ground reaction force|1|float64|
|g_v_max2_r|Right vertical push-off Ground reaction force|1|float64|
|g_v_max1_l|Left vertical landing Ground reaction force|1|float64|
|g_v_min_l|Left vertical transfer Ground reaction force|1|float64|
|g_v_max2_l|Left vertical push-off Ground reaction force|1|float64|
|AGE|Age of participant|1|int64|
|BMI|Body-mass-index of participant|1|float64|
|CLASS_LABEL|Expert labelled classification of walking gait of participant|1|string|

## 1.4. Overview ##    
[[back to the top]](#table-of-contents)

1. [Data Processing](./code/1_Data_Processing.ipynb) (By 6 May)
> * Data preparation, description and cleaning
> * Normalise each trial reading to 100% of the stride, 100 data points.
> * Normalise GRF with Body weight
2. [EDA and Feature Engineering](./code/2_EDA_Part1.ipynb) (By 13 May)
> * Visualise healthy and unhealthy walking gait
> * Create features from the normalised readings to differentiate the walking gait
> * Merge dataset
3. [More EDA and Base Line models](./code/2_EDA_Part2.ipynb) (By 16 May)
4. [Logistic Regression and Decision Tree models](./code/3_Model_Part1.ipynb) (By 18 May)
5. [Neural Network](./code/3_Model_Part2.ipynb) (By 21 May)
6. [Support Vector Machines](./code/3_Model_Part3.ipynb)(By 22 May)

**Machine Learning Models**
1. Decision Tree
2. Neural Network
3. Support Vector Machines

**Success Metric**
1. Interpretable Gait Classification model with good ROC curve and accuracy
2. List of key features as markers for each of the gait disorders

<br>

# 2. Results and learnings
[[back to the top]](#table-of-contents) <br>

## 2.1. Training and evaluation results ##
[[back to the top]](#table-of-contents) <br>

## 2.2. Model Selection and Conclusion ##
[[back to the top]](#table-of-contents) <br>

**Summary of model scores:**

**Conclusion:**

<br>

**Future Exploration:** <br>

## 3. References ##
[[back to the top]](#table-of-contents) <br>
[1] Brian Horsak et al. “GaitRec, a large-scale ground reaction force dataset of healthy and impaired gait (2020) https://www.nature.com/articles/s41597-020-0481-z

<br>


