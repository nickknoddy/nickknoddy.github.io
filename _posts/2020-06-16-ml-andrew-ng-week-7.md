---
title: "Machine Learning By Andew Ng - Week 7"
---

# Large Margin Classification

## Optimisation Objective

- Alternative View Of Logistic Regression

  - If y = 1, h(x) should be similar to 1, theta^T x >> 0

  - If y = 0, h(x) should be similar to 0, theta^T x >> 1

  ![Alternate View LOR.png](/assets/images/ml-andrew-ng-week-7/alternative-view-lor.png)

  - Graph plotting of function z

  ![Alternate View LOR 1.png](/assets/images/ml-andrew-ng-week-7/alternative-view-lor-1.png)

- Support Vector Machine

  - Modified hypothesis of Logistic Regression

  ![SVM.png](/assets/images/ml-andrew-ng-week-7/svm.png)
  ![SVM Hypothesis.png](/assets/images/ml-andrew-ng-week-7/svm-hypothesis.png)

## Large Margin Intuition

- Concept

  ![SVM Concept.png](/assets/images/ml-andrew-ng-week-7/svm-concept.png)

- SVM Decision Boundary

  ![SVM Decision Boundary.png](/assets/images/ml-andrew-ng-week-7/svm-decision-boundary.png)

  - SVM creates a decision boundary in a way that there is some space between the samples and decision boundary

  - That space is called as margin

  ![SVM Decision Boundary 1.png](/assets/images/ml-andrew-ng-week-7/svm-decision-boundary-1.png)

  - If C is too large the decision boundary will identical to magenta line

  - If C is not to large then the decision boundary will be identical to black line

  ![SVM Decision Boundary 2.png](/assets/images/ml-andrew-ng-week-7/svm-decision-boundary-2.png)

## Mathematics Behind Large Margin Classification

- Vector Inner Product

  - || u || → euclidean length of vector u

  - p → length of projection of v onto u

  ![Vector Inner Product.png](/assets/images/ml-andrew-ng-week-7/vector-inner-product.png)

- SVM Decision Boundary

  ![SVM DB.png](/assets/images/ml-andrew-ng-week-7/svm-db.png)
  ![SVM DB 1.png](/assets/images/ml-andrew-ng-week-7/svm-db-1.png)
  ![SVM DB 2.png](/assets/images/ml-andrew-ng-week-7/svm-db-2.png)

# Kernels

## Kernels I

- Non Linear Decision Boundary

  ![Non Linear DB.png](/assets/images/ml-andrew-ng-week-7/non-linear-db.png)

- Kernels

  ![Kernel.png](/assets/images/ml-andrew-ng-week-7/kernel.png)

  - Similarity

    - Gaussian Kernel

    ![Kernels and Similarity.png](/assets/images/ml-andrew-ng-week-7/kernels-and-similarity.png)

  - Exmaple

    ![Example.png](/assets/images/ml-andrew-ng-week-7/example.png)

  - Concept

    ![Concept.png](/assets/images/ml-andrew-ng-week-7/concept.png)

## Kernels II

- Choosing Landmarks

  ![Choosing Landmarks.png](/assets/images/ml-andrew-ng-week-7/choosing-landmarks.png)

- SVM with Kernels

  ![SVM with Kernels.png](/assets/images/ml-andrew-ng-week-7/svm-with-kernels.png)
  ![SVM with Kernels 1.png](/assets/images/ml-andrew-ng-week-7/svm-with-kernels-1.png)

- SVM Parameters

  ![SVM Parameters.png](/assets/images/ml-andrew-ng-week-7/svm-parameters.png)

# SVMs in Practice

## Using An SVM

- Overview

  ![Overview.png](/assets/images/ml-andrew-ng-week-7/overview.png)

- Octave Implementation

  ![Octave Implementation.png](/assets/images/ml-andrew-ng-week-7/octave-implementation.png)

- Other Kernels

  ![Other Kernels.png](/assets/images/ml-andrew-ng-week-7/other-kernels.png)

- Multi Class Classification

  ![Multi-class Classification.png](/assets/images/ml-andrew-ng-week-7/multi-class-classification.png)

- Logistic Regression VS Support Vector Machine

  ![LOR vs SVM.png](/assets/images/ml-andrew-ng-week-7/lor-vs-svm.png)

# Lecture Presentation

<embed src="/assets/pdfs/ml-andrew-ng-week-7/ml-andrew-ng-week-7.1.pdf" width="100%" height="600px" type="application/pdf">
