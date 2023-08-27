# Rider-Driven-cancellation-prediction

The challenge was to develop a model that could predict rider-driven cancellation in advance (i.e., before getting marked as cancelled or delivered) based on the order and rider details.
The data was _highly skewed_ and filled with a lot of _NaN values_ which we tried to fill by the **Progressive KNN method** we found in a Kaggle discussion, which worked well for its authors([reference link](https://www.kaggle.com/c/now-you-are-playing-with-power/discussion/300903)).


I added some features because we were left with very few of them after dropping the useless ones:
* The time difference between order time and allot time
* The time difference between order time and accept time
* The day of the week
* The hour of the day
* If it's the start of the month (Boolean)

This improved our roc_auc by 0.95, _code in rider-pred-data-cleaning.ipynb_

After exploring the data, we trained various models like XGBClassifier, CatboostClassifier, KNNClassifier, etc., and also their ensembles but the roc_auc_score was always less than 0.65. So, we tried AutoXGB on our data, which boosted our score to 0.77 on the leaderboard.

I further tried smoting (a very useful method for balancing skewed data), but it was overfitting the data (gave roc_auc of 0.91 in validation and 0.7 in lb)

Lastly, I tried stacking around 20 AutoXGB models but unfortunately, it didn't work for me, and I was left with no time to try that again.
