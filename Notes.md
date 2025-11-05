# Information about the found papers from the Seminar work


## Summary of methods with Github implementations and Datasets:

Top papers:
- **FairSTG**: 
    - no implementation could be found
    - asked author for implementation --> no answer
    - Interesting because method is model-independent and bias-generating attributes can be unknown a-priori
    - focuses on spatial-temporal forecasting, uses datasets from traffic and electricity domain
    - does not focus on sub-group fairness, focuses on fairness between individual data points (uses the MAE-variance as fairness measure)


Useful papers:
- **SA-Net**: 
    - no implementation could be found
    - asked author for implementation --> no answer
    - straightforward approach, focuses on traffic domain
    - Chicago TNC dataset is accessible, but the data is enhanced with demographic information (the merged dataset of traffic and social data is not available)
    - fairness between demographic groups should be ensured (MPE gap between these groups is used as the fairness metric)


- **STEER**:
    - https://github.com/weiliao97/Learning_Time_Series_EHR 
    - complex architecture, for general use
    - MIMIC and eICU dataset are used
    - focus on topic that sensitive information shall not be predictable of medical time series data
    - do not use a specific fairness metric, they suggest fairness is ensured when sensitive information can not be predicted


- **CA-GAN**: 
    - https://github.com/nic-olo/CA-GAN
    - synthetic data generation, for general use
    - focuses on creating synthetic data for underrepresented groups
    - MIMIC dataset is used
    - no modular implementation
    - focus on representation bias, measures fairness by deviation of subgroup accuracy

- **HiMoE**: 
    - no implementation could be found, maybe ask authors
    - complex architecture, maybe only for large datasets
    - use traffic data, fairness between sensors should be ensured
    - use standard deviation of MAE as fairness metric

- **BAHT**: 
    - they stated they wanted to publish implementation, but nothing could be found
    - only simple strategies, but it would be interesting to see how they perform 
    - ECG and EEG datasets, no very informative metadata, they use the anonymized classes as protected variables
    - fairness is only subtopic in their work

- **FairFor**:
    - https://github.com/hh4ts/FairFor/tree/main
    - work focus on fairnes between target variables
    - work must be adjusted for group fairness if it should be used
    - PeMS: traffic data of california, registration necessary
    - ECG5000: obtained and reduced from Physionet, seems to have no metadata information
    - SolarEnergy: no real metadata except regions, publicly available


- **Absolute Correlation Regularization**: 
    - no implementation could be found
    - straightforward approach, for general use
    - enhanced trip data with social average features for regions, groups regions, processed dataset is not available (no repo could be found)
    - fairness measure: correlation coefficent between accuracy and protected variables

- **Long Term Fairness**: 
    - https://github.com/yaoweihu/Achieving-Long-term-Fairness/tree/main 
    - only applicable on one bias generating attribute, considers long term effects and short term effects
    - only uses synthetic and semi-synthetic datasets
    - Classification task (predicts a category with time series data consisting of only two time points)
    - more focused on sequential decision system then on time series data

Not very useful papers:
- **FairTP**: 
    - private repo of jiangnanx129
    - focused on traffic data
    - HK Didi and SD Dataset: sourced from Hong Kong Government and PeMS, traffic data, they grouped sensors into regions which is the bias-generating attribute 
    - not applicable on multiple bias generating attributes  

- **Fairness criteria for LDS**: 
    - https://github.com/Quan-Zhou/Fairness-in-Learning-of-LDS?utm_source=catalyzex.com
    - only for linear dynamical systems, two simple regularization terms
    - synthetic data and COMPAS (typical fairness dataset), not a real time series dataset but they transformed it to one 
    - use of commerical Software Mosek in their model 

- **FairST**: 
    - https://github.com/equitensor/EquiTensor_2021/tree/master 
    - focusing on traffic domain, already surpassed by FairTP in their work
    - bike renting dataset enhanced with socioeconomic, weather and urban features, must be rebuilt as github repo does not contain the data


## Time Series Benchmarks

Time Series Benchmarks:
- https://huggingface.co/datasets/autogluon/fev_datasets
- https://github.com/GestaltCogTeam/BasicTS/tree/master/datasets
- https://github.com/thuml/Autoformer
- https://github.com/thuml/Time-Series-Library
- https://decisionintelligence.github.io/OpenTS/OpenTS-Bench/datasets/
- https://github.com/DescartesResearch/ForecastBenchmark


## Potential Useful Methods, Datasets, Metrics
Potential Methods:
- Long-Term Fairness --> Focus on fair decisions over time in sequential decision systems, do not use time series data, designed for classification tasks, contains two other interesting baselines (Fair Model with Demographic Parity/Equal Opportunity)
- CA-GAN  --> very complex, no modular implementation, MIMIC dataset is necessary
- FairST --> very complex, no modular implementation, used dataset is not available
- FairFor --> designed for MultivariateForecasting and fairness between target variables
- Fairness in LDS --> is hardly realizable, do not understand concept of LDS, not clear what is predicted and what is used as test data
- STEER  --> very complex, no modular implementation, MIMIC dataset is necessary

- currently no answer from authors of FairSTG and SA-Net


Current Datasets:
- Saudi Hospital https://www.kaggle.com/datasets/datasetengineer/king-saud-medical-city-ksmc 
- https://www.kaggle.com/datasets/iamsouravbanerjee/inequality-in-education-around-the-world education measures for countries(countries can be grouped)
- https://www.kaggle.com/datasets/akhilchhh/cosgdd (multiple measures for countries over some years, countries can be grouped)
- Restaurant visitor forecast (https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data) (data of a kaggle competition but also allowed for academic research)
- walmart dataset (https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
- Favorita_transactions (https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) 
- exchange rate dataset of LSTNet work (https://arxiv.org/pdf/1703.07015, https://drive.google.com/drive/folders/1nuMUIADOc1BNN-uDO2N7zohLgpLDgl-Z)
- Illness (https://arxiv.org/pdf/2106.13008)

Potential Datasets:
- Chicago TNC trip record or any other traffic dataset for demand prediction with fairness between regions (look at BasicTS references)
- MIMIC or eICU --> I can't access the data, compliance course is necessary (price 129$)
- Data from Uniklinkum Freiburg?
- Maybe find data from census https://www.census.gov/econ_datasets/ 
- Find further datasets from fev-bench (https://huggingface.co/datasets/autogluon/fev_datasets) or other benchmarks
- other potential dataset (traffic and weather, but need further preprocessing https://arxiv.org/pdf/2106.13008)



Potential Metrics:
- error Variance between different protected groups or individuals

potential new paper: 
https://openreview.net/forum?id=D4r8LpZshO