---
date: 2024-11-05 12:26:40
layout: post
title: Classification-Handwritten Digits
subtitle: A Comparative Study of Digit Classification Algorithms
description: KNN/SGD achieves 97% on MNIST dataset. Multilabel/output classification explored
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1731812902/classification_cover_page_2.jpg
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/classification_cover_page_2
category: Data Science
tags:
  - ML
  - Classification
  - Model Evaluation
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

**Executive Summary:** This project aims to classify handwritten digits (0-9) using machine learning techniques, primarily focusing on the MNIST(Modified National Institute of Standards and Technology database) dataset. The project explores various classification algorithms, evaluates their performance using metrics like accuracy, precision, recall, and F1-score, and implements techniques like cross-validation and hyperparameter tuning to optimize model performance. It also delves into more advanced concepts such as multilabel and multioutput classification.

The complete Python code and required file for this analysis is available on my <b><a href="https://github.com/VijaikumarSVK/Classification-Handwritten-Digits">GitHub</a></b>

### Data and Initial Setup
#### MNIST Dataset Introduction
The **MNIST** dataset, a cornerstone in image recognition, comprises 70,000 grayscale images of handwritten digits (0-9), each measuring 28x28 pixels. These images, contributed by high school students and US Census Bureau employees, serve as a robust benchmark for evaluating classification models. Each image is labeled with the digit it represents.

```js
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame = False)
```
Data exploration reveals the dataset's structure: **70,000 samples,** each with **784 features** (representing the pixel values) and corresponding labels.<br>

Below sample image represents digit 5
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731814161/some_digit_plot.png)

Visualizing a sample image helps understand the data format and verify the label accuracy.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731814490/more_digits_plot.png)
Displaying a grid of images provides a comprehensive overview of the dataset's diversity.

#### Binary Classification with SGD
As a starting point, the project simplifies the classification problem to a binary classification task: identifying the digit "5". This creates a "5-detector" that distinguishes between two classes: "5" and "not-5". The Stochastic Gradient Descent (SGD) classifier, well-suited for large datasets, is employed for this task.

```js
y_train_5 = (y_train =='5') // true for all 5's, False for all other digits
y_test_5 = (y_test == '5')
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit]) // Output -> array([ True])
```

### Model Evaluation and Performance
#### Cross-Validation for Accuracy Measurement
Cross-validation, a robust evaluation technique, is used to assess the model's performance. It involves partitioning the training set into folds, training the model on different combinations of these folds, and evaluating its performance on the held-out fold. This provides a more reliable estimate of the model's generalization ability.


```js
//cross-validation and calculate accuracy
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')
// Output -> array([0.95035, 0.96035, 0.9604 ])
```

#### Confusion Matrix Analysis
Confusion matrices offer a detailed breakdown of model predictions, highlighting the counts of true positives, true negatives, false positives, and false negatives. They provide valuable insights into the types of errors the model is making.

```js
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
// Output
// array([[53892,   687],
//        [ 1891,  3530]], dtype=int64)
```
 The first row of this matrix considers non-5 images (the negative class): 53,892 of them were correctly classified as non-5s (they are called **true negatives**), while the remaining 687 were wrongly classified as 5s (**false positives**, also called type I errors). The second row considers the images of 5s (the positive class): 1,891 were wrongly classified as non-5s (**false negatives**, also called type II errors), while the remaining 3,530 were correctly classified as 5s (**true positives**).

#### Precision and Recall Metrics
Precision and recall are crucial metrics for evaluating a classifier's performance, especially in imbalanced datasets. Precision measures the accuracy of positive predictions, while recall quantifies the model's ability to identify all positive instances.
```js
//Precision score
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) == 3530 / (687 + 3530)
// Output -> 0.8370879772350012
```
```js
//Recall score
recall_score(y_train_5, y_train_pred)# == 3530 / (1891 + 3530)
// Output -> 0.6511713705958311
```
The F1-score, the harmonic mean of precision and recall, provides a single metric that balances both aspects of performance.<br>

**F1 = 2 × ((precision × recall)/(precision + recall))**

```js
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)
// Output -> 0.7325171197343846
```

#### Precision-Recall Trade-off
There's often an inverse relationship between precision and recall. Increasing the prediction threshold can improve precision at the expense of recall, and vice-versa. Analyzing this trade-off is essential for selecting an optimal threshold based on the application's specific requirements.
```js
threshold = 3000
y_some_digit_pred = (y_scores>threshold)
y_some_digit_pred
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```
```js
plt.figure(figsize=(8,4))
plt.plot(thresholds, precisions[:-1], "b--", label = 'Precision', linewidth = 2)
plt.plot(thresholds, recalls[:-1], "g-", label  ='Recall', linewidth = 2)
plt.vlines(threshold, 0,1.0,"k", "dotted", label = "threshold")

idx = (thresholds >= threshold).argmax()
plt.plot(thresholds[idx], precisions[idx],'bo')
plt.plot(thresholds[idx], recalls[idx],"go")
plt.axis([-50000,50000,0,1])
plt.grid()
plt.xlabel('Threshold')
plt.legend(loc = "center right")
save_fig("precision_recall_vs_threshold_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731816754/precision_recall_vs_threshold_plot.png)

Another way to select a good precision/recall trade-off is to plot precision directly against recall
```js
import matplotlib.patches as patches
plt.figure(figsize=(6,5))
plt.plot(recalls, precisions, linewidth = 2, label = 'Precision/Recall curve')
plt.plot([recalls[idx], recalls[idx]],[0.,precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko", label = "Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color = "#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0,1,0,1])
plt.grid()
plt.legend(loc = 'lower left')
save_fig("precision_vs_recall_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731817186/precision_vs_recall_plot.png)

```js
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision] // --> 3370.0194991439557
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90) // --> 0.9000345901072293
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90) // --> 0.4799852425751706
```
We have 90% precision classifier, but high precision classifier is not very usefull if its recall is too low. for many application 48% recall wouldn't be great at all

#### ROC Curve and AUC
The Receiver Operating Characteristic (ROC) curve visually represents the trade-off between the true positive rate (recall) and the false positive rate. It helps assess the classifier's performance across different thresholds and compare different classifiers.

```js
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize = (6,5))
plt.plot(fpr, tpr, linewidth = 2, label = "ROC Curve")
plt.plot([0,1],[0,1],'k:', label = "Random Classifier's ROC curve")
plt.plot([fpr_90],[tpr_90],"ko", label = "Threshold for 90% precision")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12,0.71, "Higher\nthreshold", color = "#333333")
plt.xlabel('False Positive Rate (Fall-out)')
plt.ylabel('True Positive Rage (Recall)')
plt.grid()
plt.axis([0,1,0,1])
plt.legend(loc = 'lower right', fontsize = 13)
save_fig('roc_curve_plot')
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731822468/roc_curve_plot.png)

The area under the ROC curve (AUC) provides a single metric summarizing the classifier's overall performance.
```js
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores)
// Output -> 0.9604938554008616
```
comparing **RandomForestClassifier** PR and F1 score with **SGDClassifier**
```js
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = 'predict_proba')
y_probas_forest[:2]
// Output:
// array([[0.11, 0.89],
//        [0.99, 0.01]])
```
By looking the probabilities for the first two images in the training set Model predicts the first image is positve with <b>89%</b> probability, and it predicts the second image is negative with <b>99%</b> probability.

```js
y_scores_forest = y_probas_forest[:,1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize = (6,5))
plt.plot(recalls_forest, precisions_forest, "b-", linewidth = 2,
        label = 'Random Forest')
plt.plot(recalls, precisions, '--', linewidth=2, label = 'SGD')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("pr_curve_comparison_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731823975/pr_curve_comparison_plot.png)
```js
y_train_pred_forest = y_probas_forest[:,1] >=0.5
f1_score(y_train_5, y_train_pred_forest)
// Output -> 0.9274509803921569

roc_auc_score(y_train_5, y_scores_forest)
// Output -> 0.9983436731328145
```

### Multiclass and Advanced Classification
#### Multiclass Classification Techniques
Moving beyond binary classification, the project explores multiclass classification, where the goal is to classify digits into ten classes (0-9). Strategies like One-vs-Rest (OvR) and One-vs-One (OvO) are employed to handle multiclass problems.
```js
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000],y_train[:2000])
ovr_clf.predict([some_digit])
// Output -> array(['5'])
```

```js
sgd_clf = SGDClassifier(random_state= 42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
// Output -> array([0.87365, 0.85835, 0.8689 ])
```

#### Error Analysis and Visualization
Analyzing the types of errors made by the classifier is crucial for improving its performance. Confusion matrices are invaluable for this purpose. Normalizing the confusion matrix allows for a clearer understanding of error patterns.<br>

The darker cell for digit '5' in the confusion matrix suggests either more errors on 5s or fewer 5s in the dataset. Normalizing (by row) addresses this imbalance.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731825442/confusion_matrix_plot_1.png)

Only 82% of 5s were correctly classified, most often misclassified as 8s (10% of 5s vs. 2% of 8s). Highlighting errors is possible by zeroing-out correct predictions' weights.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731825539/confusion_matrix_plot_2.png)

It is also possible to normalize the confusion matrix by column rather than by row. if you set normalize="pred", you get the diagram on the right in Figure For example, you can see that 56% of misclassified 7s are actually 9s.<br>

Analyzing individual errors can also be a good way to gain insights into what your
classifier is doing and why it is failing.

```js
cl_a, cl_b = '3','5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

size = 5
pad = 0.2
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")
save_fig("error_analysis_digits_plot")
plt.show()       
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731827260/error_analysis_digits_plot.png)

#### Multilabel Classification
Multilabel classification allows a single instance to be assigned to multiple classes simultaneously. For instance, a digit image could be labeled as both "large" (7, 8, or 9) and "odd".

```js
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="weighted")
// Output -> 0.9778357403921755
```

#### Multioutput Classification
Multioutput classification generalizes multilabel classification by allowing each label to be multiclass. This project demonstrates this concept by building a system to remove noise from digit images.

```js
np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
save_fig("noisy_digit_example_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731827284/noisy_digit_example_plot.png)

let's train the classifier and make it clean up this image

```js
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
save_fig("cleaned_digit_example_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731826423/cleaned_digit_example_plot.png)


### Model Optimization and Results
#### Accuracy Evaluation
Evaluating the model's accuracy on a held-out test set is crucial for assessing its real-world performance.
```js
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
baseline_accuracy = knn_clf.score(X_test, y_test)
// Output -> 0.9688
```

#### Hyperparameter Tuning

Hyperparameter tuning aims to optimize the model's performance by systematically searching through the space of possible hyperparameter values. Techniques like GridSearchCV can automate this process.

```js
from sklearn.model_selection import GridSearchCV
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train[:10_000], y_train[:10_000])
grid_search.best_score_
// Output -> 0.9441999999999998
```
Score was dropped to 94% when we only trained 10000 image.

#### Achieving 97% Accuracy

By tuning hyperparameters and training on the full dataset, the project achieves an improved accuracy of 97%.

```js
grid_search.best_estimator_.fit(X_train, y_train)
tuned_accuracy = grid_search.score(X_test, y_test)
Output -> 0.9714
```

This project demonstrates a comprehensive approach to classifying handwritten digits, covering various machine learning techniques, evaluation metrics, and optimization strategies. The combination of exploring different algorithms, fine-tuning hyperparameters, and analyzing errors leads to a highly accurate model capable of achieving 97% accuracy on the MNIST dataset.
