<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Conformal Predictor: a handy script for prediction uncertainty quantification

Author: Xianghao Zhan, Li Liu

Date: 3/30/2023

## Goal
In addition to making predictions, uncertainty quantification is also important for machine learning classification models. This tool will leverage conformal prediction to help researchers compute the reliability of the predictions with the predicted probabilities output by any machine learning classifier. Users only need to input 1) the training labels with the predicted probabilities for each label; 2) the predicted labels for the test data with the predicted probabilities for each label, the script will output the credibility and confidence of the prediction. Users can report either/both of them in your manuscript as the quantified reliability for the classification predictions.

## Input
train.csv: N rows (N samples), D+1 columns (D possible labels); csv file;  first column is the training label (int, coded from 0 to D-1), the remaining D columns are the predicted probabilities for each of the D possible labels given by your machine learning classifier [the row sum should be 1].

test.csv: N rows (N samples), D+1 columns (D possible labels); csv file;  first column is the predicted label (int, coded from 0 to D-1), the remaining D columns are the predicted probabilities for each of the D possible labels given by your machine learning classifier [the row sum should be 1].

## Output
Credibility: how much reliability we have in selecting the current predicted label (the largest p-value)

Confidence: how much reliability we have in rejecting the other possible labels (1 - the second largest p-value)

## How to use this script?
1. Prepare the two csv files

2. Download the .py file or the .exe file

3. Method 1) Open a shell/command line window at the directory where the three files are and input: 

`
python Conformal_Prediction.py {train.csv} {test.csv} {D:the number of possible label}
`

   Method 2) Opeh a shell/command line window at the directory where the three files are and input: 
   
   `
   Conformal_Predictor.exe {train.csv} {test.csv} {D:the number of possible label}
   `

Please cite our work if you use this handy script as an add-on to your machine learning projects! Please reach out to Xianghao for potential suggestions and collaborations: xzhan96@stanford.edu

## Reference
Publications that use the current method to compute non-conformity measurement to quantify uncertainty:

[1] Liu L, Zhan X, Yang X, Guan X, Wu R, Wang Z, Luo Z, Wang Y, Li G. CPSC: Conformal prediction with shrunken centroids for efficient prediction reliability quantification and data augmentation, a case in alternative herbal medicine classification with electronic nose. IEEE Transactions on Instrumentation and Measurement. 2022 Jan 3;71:1-1.

[2] Zhan X, Wang F, Gevaert O. Reliably Filter Drug-Induced Liver Injury Literature With Natural Language Processing and Conformal Prediction. IEEE Journal of Biomedical and Health Informatics. 2022 Jul 25;26(10):5033-41.

[3] Xu C, Xu Q, Liu L, Zhou M, Xing Z, Zhou Z, Zhou C, Li X, Wang R, Wu Y, Wang J, Zhang L, Zhan X, Gevaert O, Lu G. A Tri-light Warning System for Hospitalized COVID-19 Patients: Credibility-based Risk Stratification under Data Shift. medRxiv. 2022:2022-12.

[4] Wang H, Zhan X, Liu L, Ullah A, Li H, Gao H, Wang Y, Hu R, Li G. Unsupervised cross-user adaptation in taste sensation recognition based on surface electromyography. IEEE Transactions on Instrumentation and Measurement. 2022 May 18;71:1-1.

Publications related to conformal predictor for reliability quantification in general:

[1] Liu L, Zhan X, Wu R, Guan X, Wang Z, Zhang W, Pilanci M, Wang Y, Luo Z, Li G. Boost AI power: Data augmentation strategies with unlabeled data and conformal prediction, a case in alternative herbal medicine discrimination with electronic nose. IEEE Sensors Journal. 2021 Aug 3;21(20):22995-3005.

[2] Zhan X, Wang Z, Yang M, Luo Z, Wang Y, Li G. An electronic nose-based assistive diagnostic prototype for lung cancer detection with conformal prediction. Measurement. 2020 Jul 1;158:107588.

[3] Zhan X, Guan X, Wu R, Wang Z, Wang Y, Luo Z, Li G. Online conformal prediction for classifying different types of herbal medicines with electronic nose.

[4] Zhan X, Guan X, Wu R, Wang Z, Wang Y, Li G. Discrimination between alternative herbal medicines from different categories with the electronic nose. Sensors. 2018 Sep 4;18(9):2936.

## What does the code do?
To compute the credibility and confidence, we leverage the conformal predictor based on the prediction probability and the steps can be summarized as the following steps:

1) Convert the predicted probability to a nonconformity measurement: a metric to quantify how well a particular feature-label combination conforms to the training data. Here, we leveraged a design of the nonconformity measurement $\alpha_i$ that has been validated in previous machine learning applications [1,2,4,5]:

$$
\alpha_i=0.5-\frac{\hat{p}\left(y_i \mid x_i\right)-\max \hat{p}_{y!=y_i} \left(y_{i} \mid x_{i}\right)}{2}
$$

Here $y_i$ and $x_i$ denotes the label and feature of the $i-th$ sample. The predicted probability can be computed by any classifier that can output predicted probability: MLP, LR, RF, GBM, LDA.

2) Based on the nonconformity measurement, all the training samples' nonconformity measurement values can be computed and the distribution will be further used to calibrate the reliability we have for a new prediction;

3) When making a new prediction, the nonconformity measurement $\alpha*$ for the test sample $x*$ is computed based on the previous equation. Then, the P-value of the prediction, which indicates the credibility of the prediction is calculated by investigating the fraction of samples in the training distribution with larger nonconformity measurement.

4) The credibility of the prediction can be computed as the larger P-value, which reflects how well the most likely label conform to the distribution of the training data nonconformity measurement. The confidence is 1 - the second largest P-value, which reflects how well the other possible labels (excluding the predicted labels) conform to the distribution of the training data nonconformity measurement. The two metrics may provide us with a flexible tool to adapt the decision.
