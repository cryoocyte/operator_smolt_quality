import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (y_pred + y_true)) * 100

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ModelPipeline:
    def __init__(self, data, features, target, model):
        self.data = data
        self.features = features
        self.target = target
        self.model = model
        self.cv_parameters = {
            'nthread': [4],
            'objective': ['reg:linear'],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [5, 6, 7],
            'min_child_weight': [4],
            'silent': [1],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'n_estimators': [500],
            'verbosity': [0],
        }

    def split_data(self, test_size=0.2, random_state=1):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data[self.features],
            self.data[self.target],
            test_size=0.2,
            random_state=1
        )

    def cv_model(self, parameters=None, scoring='neg_mean_absolute_percentage_error'):
        grid = GridSearchCV(
            self.model(),
            parameters or self.cv_parameters,
            cv=2,
            n_jobs=5,
            scoring=scoring,
            verbose=False
        )
        grid.fit(self.X_train.to_numpy(), self.y_train.to_numpy())

        print(grid.best_score_)
        print(grid.best_params_)

        self.model_best = self.model(**grid.best_params_)

    def evaluate_model(self, prefix='xgb_'):
        self.model_best.fit(self.X_train, self.y_train)
        self.y_test_pred = self.model_best.predict(self.X_test)
        smape_val = smape(self.y_test, self.y_test_pred)
        self.smape_val = smape_val if smape_val > 0 else smape(np.exp(self.y_test), np.exp(self.y_test_pred))
                
        self.y_train_pred = self.model_best.predict(self.X_train)
        smape_train = smape(self.y_train, self.y_train_pred)
        self.smape_train = smape_train if smape_train > 0 else smape(np.exp(self.y_train), np.exp(self.y_train_pred))
        
#         display('r2:', r2_score(self.y_test, self.y_test_pred))
#         display('MAPE:', mean_absolute_percentage_error(self.y_test, self.y_test_pred))
#         display('MSE:', mean_squared_error(self.y_test, self.y_test_pred))
        
        self.importance = self.model_best.feature_importances_
        self.model_factors = pd.Series(self.importance, index=self.features)

        plt.figure(figsize=(15, 7))
        fig = self.model_factors.sort_values(ascending=True).plot(kind='barh')
#         fig.get_figure().savefig(f'{prefix}model_factors_importance.jpg' )

        self.factors_target_corr = self.data[self.model_factors.index]\
        .corrwith(self.data[self.target])
        
    def process(self):
        self.split_data()
        self.cv_model()
        self.evaluate_model()
