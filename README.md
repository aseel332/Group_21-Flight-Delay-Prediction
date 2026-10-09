# Flight Delay Prediction

A machine learning pipeline that predicts whether a flight will arrive **45+ minutes late** (classification) and **by how many minutes** (regression), trained on ~484,000 U.S. flight records. Built as a group project for ECS 171 (Machine Learning) at UC Davis.

## Results

| Task | Best model | Test performance |
| --- | --- | --- |
| Classification: severe delay (45+ min) | MLP neural network | **86.4% accuracy, 0.926 ROC-AUC** |
| Regression: delay in minutes | Tuned Random Forest | **R² 0.715**, MAE 19.0 min, RMSE 30.9 min |

Models compared: Linear SVM, MLP, Random Forest (classification); Linear Regression, Decision Tree, Random Forest (regression).

## Approach

1. **Data:** the [Flight Delay and Causes](https://www.kaggle.com/datasets/undersc0re/flight-delay-and-causes) dataset on Kaggle, about 500K flights and 28 features.
2. **EDA:** arrival delay is strongly right-skewed, and numeric features correlate weakly with it. Delay rates vary much more by airline and airport, so categorical features carry most of the signal.
3. **Feature engineering:** derived month, day, departure hour/minute and scheduled-arrival hour/minute from raw timestamps. Dropped `DepDelay` to avoid leakage, since it is almost perfectly correlated with the target.
4. **Encoding and scaling:** one-hot encoding of airline, origin and destination (570 features). Standard scaling for SVM and MLP.
5. **Training and tuning:** stratified train/test split, then hyperparameter search on a held-out validation set, with train-vs-test checks for over- and underfitting.

## Key findings

- The MLP beat the linear SVM (61.8% accuracy) by learning non-linear interactions between time, airline and route.
- The default Random Forest overfit badly (99.99% train vs 74.8% test accuracy). Tuning closed the gap.
- For regression, tuning the Random Forest traded a little raw test accuracy for much better generalization.
- Biggest improvement opportunity: richer features like weather, airport congestion and aircraft rotation.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter / Google Colab, joblib

## Run it

1. Download `Flight_delay.csv` from the Kaggle link above and place it next to the notebook.
2. Open `Flight_delay_predictor.ipynb` in Jupyter or Colab.
3. Set `TRAIN_MODELS = True` to train from scratch, or keep it `False` to load saved artifacts from `flight_delay_artifacts.joblib`.
