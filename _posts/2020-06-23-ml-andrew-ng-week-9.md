---
title: "Machine Learning By Andew Ng - Week 9"
---

# Density Estimation

## Problem Motivation

- Anomaly Detection Example

  - Plotting the dataset, and compare it with the new datapoint for its behaviour

  - If its in the same range as the dataset then the new datapoint is identified as ok

  - It its not in the same range as the dataset then the new datapoint is flagged as anomaly

  ![Anamoly Detection Example.png](/assets/images/ml-andrew-ng-week-9/anomaly-detection-example.png)

- Density Estimation

  - If new datapoint is less than some value ( epsilon ) then flagged as anomaly

  - If new datapoint is equal to or more than some value ( epsilon ) then identified as ok

  ![Density Estimation.png](/assets/images/ml-andrew-ng-week-9/density-estimation.png)

- Anomaly Detection Applications

  - Fraud Detection

  - Manufacturing

  - Monitoring computers in a data center

  ![Anamoly Detection Applications.png](/assets/images/ml-andrew-ng-week-9/anomaly-detection-applications.png)

## Gaussian Distribution

- Gaussian Distribution

  - Say x belongs to Real Number.

  - If x is a distributed Gaussian with mean and variance

  ![Gaussian Distribution.png](/assets/images/ml-andrew-ng-week-9/gaussian-distribution.png)

- Gaussian Distribution Examples

  ![Gaussian Distribution Example.png](/assets/images/ml-andrew-ng-week-9/gaussian-distribution-example.png)

- Parameter Estimation

  - Finding mean and variance from the Gaussian Distribution

  ![Parameter Estimation.png](/assets/images/ml-andrew-ng-week-9/parameter-estimation.png)

## Algorithm

- Density Estimation

  - Big notation of pi indicates product

  ![DE.png](/assets/images/ml-andrew-ng-week-9/de.png)

- Algorithm

  - Choose features x, that might be indicative of anomalous examples

  - Fit parameters - mean and variance

  - Given new example x, compute p( x )

  - p ( x ) ≥ epsilon ⇒ OK

  - p ( x ) < epsilon ⇒ Anomaly

  ![Algorithm.png](/assets/images/ml-andrew-ng-week-9/algorithm.png)

- Anomaly Detection Example

  ![AD Example.png](/assets/images/ml-andrew-ng-week-9/ad-example.png)

# Building an Anomaly Detection System

## Developing and Evaluating an Anomaly Detection System

- Importance of Real Number Evaluation

  - When developing a learning algorithm ( choosing features, etc . ) making decisions is much easier if we have a way of evaluating our learning algorithm

  ![Importance of real number Evaluation.png](/assets/images/ml-andrew-ng-week-9/importance-of-real-number-evaluation.png)

- Data Split

  - 60 - 20 - 20 data split

  ![Data Split.png](/assets/images/ml-andrew-ng-week-9/data-split.png)

- Evaluation

  - Evaluation metrics

    - True positive, false positive, false negative, true negative

    - Precision / Recall

    - F score

  - Use cross validation set to choose parameter epsilon

  ![Evaluation.png](/assets/images/ml-andrew-ng-week-9/evaluation.png)

## Anomaly Detection vs Supervised Learning

- Anomaly Detection vs Supervised Learning

[//]: # "child_database is not supported"

![AD vs SL.png](/assets/images/ml-andrew-ng-week-9/ad-vs-sl.png)

- Anomaly Detection vs Supervised Learning Examples

[//]: # "child_database is not supported"

![AD vs SL Example.png](/assets/images/ml-andrew-ng-week-9/ad-vs-sl-examples.png)

## Choosing What Features to Use

- Non Gaussian Features

  - Transform your non gaussian data to the form of gaussian by doing some operations on it and then feed it to the algorithm

  - It will work even if its not transformed, but it will gives less performance

  ![Non Gaussian Features.png](/assets/images/ml-andrew-ng-week-9/non-gaussian-features.png)

- Error Analysis

  - Find new features by analysing the mistake done by the algorithm in flagging anomaly

  ![Error Analysis.png](/assets/images/ml-andrew-ng-week-9/error-analysis.png)

- Example

  - Monitoring computers in a data center

    - Choose features that might take on unusually large or small values in the event of anomaly

    - x1 = memory use of computer

    - x2 = number of disk accesses / sec

    - x3 = CPU load

    - x4 = network traffic

    - x5 = CPU load / network traffic ( new feature )

  ![Example.png](/assets/images/ml-andrew-ng-week-9/example.png)

# Multivariate Gaussian Distribution

## Multivariate Gaussian Distribution

- Motivating Example

  - Monitoring machines in a data center

  ![Motivating Example.png](/assets/images/ml-andrew-ng-week-9/motivating-example.png)

- Multivariate Gaussian Distribution

  - x belongs to real number

  - Don't model p ( x1 ), p ( x2 ) .... separately.

  - Model p ( x ) all in one go

  - Parameters: mean, covariance matrix

  ![Multivariate Gaussian Distribution.png](/assets/images/ml-andrew-ng-week-9/multivariate-gaussian-distribution.png)

- Multivariate Gaussian Distribution Examples

  - Covariance matrix is altered

  - Only the first diagonal

  - Even alteration

  ![MGD Example 1.png](/assets/images/ml-andrew-ng-week-9/mgd-example-1.png)

  - Uneven alteration

  ![MGD Example 2.png](/assets/images/ml-andrew-ng-week-9/mgd-example-2.png)

  - Altering second diagonal evenly

  ![MGD Example 3.png](/assets/images/ml-andrew-ng-week-9/mgd-example-3.png)

  - Altering second diagonal values negatively

  ![MGD Example 4.png](/assets/images/ml-andrew-ng-week-9/mgd-example-4.png)

  - Altering the mean value

  ![MGD Example 5.png](/assets/images/ml-andrew-ng-week-9/mgd-example-5.png)

## Anomaly Detection using the Multivariate Gaussian Distribution

- Multivariate Gaussian Distribution

  - Formula

    - Finding the parameters with the formula

    ![MGD Formula.png](/assets/images/ml-andrew-ng-week-9/mgd-formula.png)

  - Flow

    - Substituting the values of parameters in the formula

    ![MGD FLow.png](/assets/images/ml-andrew-ng-week-9/mgd-flow.png)

  - Relationship to the original model

    - It can proved as the special case of the multivariate gaussian distribution where it aligned with the axis

    ![Relationship to the original model.png](/assets/images/ml-andrew-ng-week-9/relationship-to-original-model.png)

- Differentiation

[//]: # "child_database is not supported"

- Original Model vs Multivariate Gaussian

![Differentiation.png](/assets/images/ml-andrew-ng-week-9/differentiation.png)

# Predicting Movie Ratings

## Problem Formulation

- Example

  - Predicting Movie Rating

    - n_u ⇒ no. of users

    - n_m ⇒ no. of movies

    - r ( i , j ) = 1 ⇒if user j has rated movie i

    - y ^ ( i , j ) ⇒ rating given by the user j to movie i ( defined only if r ( i , j ) = 1 )

  ![Problem Formulation.png](/assets/images/ml-andrew-ng-week-9/probelm-formulation.png)

## Content Based Recommendations

- Content Based Recommender System

  - This is recommender system which uses one form of linear regression

  ![Content Based Recommender System.png](/assets/images/ml-andrew-ng-week-9/content-based-recommender-system.png)

  - Problem Formulation

    - r ( i , j ) = 1 ⇒if user j has rated movie i

    - y ^ ( i , j ) ⇒ rating given by the user j to movie i ( defined only if r ( i , j ) = 1 )

    - theta^j ⇒ parameter vector for user j

    - x ^ i ⇒ feature vector for movie i

    ![CBRS Problem Formulation.png](/assets/images/ml-andrew-ng-week-9/cbrs-probelm-formulation.png)

  - Optimisation Objective

    ![Optimisation Objective.png](/assets/images/ml-andrew-ng-week-9/optimisation-objective.png)

  - Gradient Descent Update

    ![Gradient Descent Update.png](/assets/images/ml-andrew-ng-week-9/gradient-descent-update.png)

# Collaborative Filtering

## Collaborative Filtering

- Problem Motivation

  ![Problem Motivation.png](/assets/images/ml-andrew-ng-week-9/problem-motivation.png)

- Optimisation Algorithm

  ![CF Optimisation Algorithm.png](/assets/images/ml-andrew-ng-week-9/cf-optimisation-algorithm.png)

- Collaborative Filtering

  - Given x, to learn theta

  - Given theta, to learn x

  - Continuous updating both values

  ![Collaborative Filtering.png](/assets/images/ml-andrew-ng-week-9/collaborative-filtering.png)

## Collaborative Filtering Algorithm

- Collaborated Formula

  - Formula has been concatenated of the earlier formula

  ![Collaborated Formula.png](/assets/images/ml-andrew-ng-week-9/collaborated-formula.png)

- Collaborative Flow

  - Initialise x and theta to small random values

  - Minimise J ( x , theta ) using gradient descent ( or any other advance optimisation algorithm )

  - For a user with parameter theta and a movie with ( learned ) features x, predict a star rating of theta^T

  ![CF Flow.png](/assets/images/ml-andrew-ng-week-9/cf-flow.png)

# Low Rank Matrix Factorisation

## Vectorisation: Low Rank Matrix Factorisation

- Collaborative Filtering

  ![CF.png](/assets/images/ml-andrew-ng-week-9/cf.png)

- Low Rank Matrix Factorisation

  ![Low Rank Matrix Factorisation.png](/assets/images/ml-andrew-ng-week-9/low-rank-matrix-factorisation.png)

- Finding Related Movies

  ![Finding Related Movies.png](/assets/images/ml-andrew-ng-week-9/finding-related-movies.png)

## Implementation Detail: Mean Normalisation

- Users who have not rated any movies

  ![Users Not Rated.png](/assets/images/ml-andrew-ng-week-9/users-not-rated.png)

- Mean Normalisation

  ![Mean Normalisation.png](/assets/images/ml-andrew-ng-week-9/mean-normalisation.png)

# Lecture Presentations

<embed src="/assets/pdfs/ml-andrew-ng-week-9/ml-andrew-ng-week-9.1.pdf" width="100%" height="600px" type="application/pdf">
<embed src="/assets/pdfs/ml-andrew-ng-week-9/ml-andrew-ng-week-9.2.pdf" width="100%" height="600px" type="application/pdf">
