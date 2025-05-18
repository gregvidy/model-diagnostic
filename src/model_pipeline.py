import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance


class ModelPipeline:
    """
    Generic pipeline for supervised classification tasks.
    """

    def __init__(self, model_type="random_forest", random_state=42):
        """
        Initialize pipeline with specified model.

        Parameters:
        -----------
        model_type : str, default='random_forest'
            Options: 'logistic_regression', 'random_forest',
            'gradient_boosting', 'xgboost', 'svm'.
        random_state : int, default=42
            Seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model(model_type)
        self.pipeline = None
        self.preprocessor = None
        self.cv_scores = None

    def _get_model(self, model_type):
        """
        Retrieve model instance by type.
        """
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight="balanced_subsample",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=1000,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
            ),
            "svm": SVC(probability=True, random_state=self.random_state),
        }
        if model_type not in models:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Choose from: {list(models.keys())}"
            )
        return models[model_type]

    def prepare_data(
        self, df, target_column, exclude_columns=None, categorical_columns=None
    ):
        """
        Prepare features X and target y for training.

        Returns:
        --------
        X : pd.DataFrame of features
        y : pd.Series target
        numerical_columns : list
        categorical_columns : list
        """
        data = df.copy()
        if target_column not in data:
            raise KeyError(f"Column '{target_column}' not in dataframe")
        y = data[target_column]
        X = data.drop(columns=[target_column] + (exclude_columns or []))

        if categorical_columns is None:
            categorical_columns = [
                col
                for col in X.columns
                if X[col].dtype == "object"
                or X[col].dtype.name == "category"
                or (X[col].nunique() < 10 and not pd.api.types.is_float_dtype(X[col]))
            ]
        numerical_columns = [col for col in X.columns if col not in categorical_columns]

        for col in numerical_columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

        for col in categorical_columns:
            X[col] = X[col].fillna("missing")

        return X, y, numerical_columns, categorical_columns

    def split_data(self, X, y, test_size=0.3):
        """
        Return dict of train/test splits.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def build_pipeline(self, numerical_columns, categorical_columns):
        """
        Construct preprocessing + classifier pipeline.
        """
        num_trans = Pipeline([("scaler", StandardScaler())])
        cat_trans = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        transformers = []
        if numerical_columns:
            transformers.append(("num", num_trans, numerical_columns))
        if categorical_columns:
            transformers.append(("cat", cat_trans, categorical_columns))
        self.preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )
        self.pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("classifier", self.model)]
        )

    def train(self, X_train, y_train, show_training_log=False):
        """
        Fit pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built.")
        if show_training_log and self.model_type == "xgboost":
            params = {
                "classifier__eval_set": [(X_train, y_train)],
                "classifier__verbose": True,
            }
            self.pipeline.fit(X_train, y_train, **params)
        else:
            self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Print accuracy, AUC, report.
        """
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(classification_report(y_test, y_pred))
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
        }

    def cross_validate(self, X, y, cv=5, scoring="roc_auc"):
        """
        Stratified CV with tqdm.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = []
        for i, (tr_idx, te_idx) in enumerate(
            tqdm(skf.split(X, y), total=cv, desc="CV folds"), 1
        ):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            self.pipeline.fit(X_tr, y_tr)
            y_prob = self.pipeline.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            tqdm.write(f"Fold {i}/{cv} AUC: {auc:.4f}")
            scores.append(auc)
        self.cv_scores = np.array(scores)
        return self.cv_scores

    def get_feature_importance(self, X, y):
        """
        Return DataFrame of importances.
        """
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            names = self.preprocessor.get_feature_names_out()
            return pd.DataFrame(
                {"feature": names, "importance": clf.feature_importances_}
            )
        perm = permutation_importance(
            self.pipeline, X, y, n_repeats=5, random_state=self.random_state
        )
        return pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})

    def get_model(self):
        """
        Return fitted classifier.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        return self.pipeline.named_steps["classifier"]

    def get_preprocessor(self):
        """
        Return fitted transformer for new data.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not set.")
        return self.preprocessor

    def get_pipeline(self):
        """
        Return entire pipeline (transformer + model).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built.")
        return self.pipeline
