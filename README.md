# Kaggle_Sirio_Libanes_ICU_Prediction
Data challenge: https://www.kaggle.com/S%C3%ADrio-Libanes/covid19

Conclusions
As will be seen throughout this work, it is part of the data scientist's life to iterate between exploratory data analysis, treatment of the database, modifications of the hyperparameters of the estimators, model training and testing;

An extensive exploratory data analysis was carried out, which resulted in the elimination of some data that would compromise the quality of the models;

A first round of machine learning was carried out for 3 different estimators: decision tree classifier, random forest and logistic regression; it was concluded, however, that the performance of such predictions represented nothing, being a mere product of luck;

To support a real evaluation of the model, cross-validation was presented and performed for other 4 training and testing rounds.

In addition to cross-validation, some hyperparameters were also varied and tested for the 3 different estimators mentioned above; the difference in predictive power, in general, varied considerably as the hyperparameter varied;

The best estimator, for the analyzes carried out in this work, was the random forest, followed by the logistic regression;

It was noticed a considerable increase in performance for all the estimators by reducing the number of database columns (features) that were duplicated and highly correlated with themselves; however, by exaggerating the dose and removing more features, we negate the benefits previously obtained;

It was diagnosed that in the initial treated database there was a data leakage that could be impacting the predictive capacity of the models; a new round of tests was carried out, but there was no success in improving the performance obtained previously;

The failure of this attempt, however, has resulted in reflection and consideration of the impact on classification analysis when data collection and data availability is already carried out in an unbalanced manner from the beggining of the work.

Next steps and strategies were discussed in an attempt to improve the predictive performance of the estimators.
