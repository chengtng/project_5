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

The target audience will be general public, carers, phyiso and occupational therapists. <br>

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

## 1.4. Overview ##    
[[back to the top]](#table-of-contents)

1. [Data Processing](./code/1_Data_Processing.ipynb)
2. EDA to be done to visualise healthy walking posture
3. EDA to be done to visualise unhealthy walking posture
4. Machine learning model training
5. Time series?


**Machine Learning Models**
1. Decision Tree
2. SVN
3. NN


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


