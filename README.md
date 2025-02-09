# NFL-Predictions
Building different machine learning models on a large dataset to predict winners of upcoming football games.

This is a personal project where I try to build models to predict the results of upcoming football games. I found a large dataset online which I am using for this task.

What makes this task challenging is the fact that the dataset is extremely complex. Using a neural network would make sense intuitively as it is more likely to fit the data better by adding enough layers and using the correct activation functions and optimizer. However making the prediction this way may not be ideal as the prediction function qould require the same number of parameters as the trained model. For my task I primarily used an RNN and the prediction function would require dummy values for home and away team scores, which defeats the whole point of making a prediction. Nonetheless, I set these values to 0 for a "fair" game. The model's peak accuracy was 71% but at this time it is around 62%. Precision was > 80% but low recall score < 45%.

I also wrote a random forest as it is handles categorical and numerical data both rather efficiently and is robust enough to be less sensitive to missing values. The downside, however, was much lower accuracy (about 30%).

The project is still a work in progress and I will continue working on it. Next steps include increasing Random Forest efficiency, increasing the parameters used to train these models to make better models, and even try other models such as XGBoost. The goal is to have good enough models before the Super Bowl.

Current predictions before today's games:
  - Neural Network: WAS beats PHI; BUF beats KC
  - Random Forest: PHI beats WAS; KC beats BUF

## Updates
9:15 PM - both the predictions made by the random forest were correct.

2/9/2-25 - the RNN continues to show higher accuracy peaking at about 68 with an updated dataset from the conference finals results. I am now using more attributes such as temperature, wind, location, and even the referee for the analysis. Random forest accuracy increased to as good as a coin flip and predicts the Chiefs to win. The RNN predicts to Eagles to win by 1 point. At this stage while the neural network has a better accuracy the random forest feels more robust and reliable. That said, I expect the Eagles to win today.
