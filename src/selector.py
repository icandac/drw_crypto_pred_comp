import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    def __init__(
        self,
        min_var_threshold=0.1,
        corr_threshold=0.95,
        univariate_pval_threshold=0.05,
        univariate_k='all',
        rf_threshold='median',
        rf_n_estimators=40,
        rf_max_features='sqrt',
        top_n_features=30,
    ):
        self.min_var_threshold = min_var_threshold
        self.corr_threshold = corr_threshold
        self.univariate_pval_threshold = univariate_pval_threshold
        self.univariate_k = univariate_k
        self.rf_threshold = rf_threshold
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_features = rf_max_features
        self.top_n_features = top_n_features

    def variance_filtering(self, X):
        variances = X.var()
        return X.loc[:, variances > self.min_var_threshold]

    def correlation_filtering(self, X):
        corr_matrix = X.corr()
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.corr_threshold:
                    to_drop.add(corr_matrix.columns[j])
        return X.drop(columns=list(to_drop))

    def univariate_feature_selection(self, X, y):
        selector = SelectKBest(score_func=f_regression, k=self.univariate_k)
        selector.fit(X, y)
        scores = selector.scores_
        pvalues = selector.pvalues_

        results = pd.DataFrame({
            'feature': X.columns,
            'f_score': scores,
            'p_value': pvalues
        }).sort_values(by='f_score', ascending=False)

        selected = results[results['p_value'] < self.univariate_pval_threshold]['feature']
        return X[selected], results

    def random_forest_selection(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            random_state=42,
            n_jobs=-1,
            max_features=self.rf_max_features
        )
        selector = SelectFromModel(rf, threshold=self.rf_threshold)
        selector.fit(X_scaled, y)

        selected_features = X.columns[selector.get_support()]
        feature_importances = selector.estimator_.feature_importances_

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)

        # Keep only top_n_features by importance
        top_features = importance_df.head(self.top_n_features)['feature'].tolist()
        return X[top_features], importance_df
