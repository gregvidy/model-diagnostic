import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    precision_score, recall_score, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin


def full_preprocess_for_tree_models(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing for tree models...")
    df_processed = df.copy()
    float_cols = df_processed.select_dtypes(include='float').columns
    print(f"Downcasting float columns: {list(float_cols)}")
    df_processed[float_cols] = df_processed[float_cols].apply(
        pd.to_numeric, downcast='float'
    )

    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            unique_vals = df_processed[col].dropna().unique()
            if set(unique_vals).issubset({'Y', 'N'}):
                print(f"Mapping Y/N to 1/0 in column: {col}")
                df_processed[col] = df_processed[col].map({'N': 0, 'Y': 1})
        if col.startswith('Is') and df_processed[col].dtype in ['object', 'int64']:
            print(f"Mapping Is* column Y/N to 1/0: {col}")
            df_processed[col] = df_processed[col].map({'N': 0, 'Y': 1})

    for col in float_cols:
        if any(k in col.lower() for k in ['amt', 'amount', 'limit']):
            print(f"Applying log10 transform to column: {col}")
            vals = df_processed[col]
            df_processed[col] = np.log10(vals + 1)

    obj_cols = df_processed.select_dtypes(include='object').columns
    print(f"Converting object columns to category: {list(obj_cols)}")
    for col in obj_cols:
        df_processed[col] = df_processed[col].astype('category')

    print("Preprocessing complete.")
    gc.collect()
    return df_processed


class ModelPipeline:
    def _init_lightgbm(self):
        import lightgbm as lgb
        print("Initializing LightGBM model...")
        return lgb.LGBMClassifier(
            n_estimators=1000,
            class_weight='balanced',
            random_state=self.random_state,
        )

    def __init__(self, model_type="xgboost", random_state=42):
        print(f"Initializing ModelPipeline with model type: {model_type}")
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model(model_type)
        self.pipeline = None
        self.preprocessor = None
        self.cv_scores = None

    def _get_model(self, model_type):
        print(f"Retrieving model instance for type: {model_type}")
        models = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=1000,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
            ),
            "lightgbm": self._init_lightgbm(),
        }
        if model_type not in models:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return models[model_type]

    def prepare_data(self, df, target_column, exclude_columns=None,
                     categorical_columns=None):
        print("Preparing data...")
        df = full_preprocess_for_tree_models(df)

        if target_column not in df:
            raise KeyError(f"Missing target column: {target_column}")

        y = df[target_column]
        X = df.drop(columns=[target_column] + (exclude_columns or []))

        if categorical_columns is None:
            categorical_columns = [
                col for col in X.columns if str(X[col].dtype) == "category"
            ]
        print(f"Identified categorical columns: {categorical_columns}")

        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        print(f"Identified numerical columns: {numerical_columns}")

        for col in numerical_columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())

        for col in categorical_columns:
            X[col] = X[col].cat.add_categories("missing").fillna("missing")

        print("Data preparation complete.")
        return X, y, numerical_columns, categorical_columns

    def split_data(self, X, y, test_size=0.3):
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=self.random_state
        )
        print("Data splitting complete.")
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def build_pipeline(self, X_train=None, y_train=None):
        print("Building pipeline...")
        class IdentityTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return X

        self.pipeline = Pipeline([
            ('identity', IdentityTransformer()),
            ('classifier', self.model)
        ])
        print("Pipeline built.")

    def train(self, X_train, y_train, show_training_log=False):
        print("Training model...")
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
        print("Model training complete.")

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(classification_report(y_test, y_pred))

        return {
            "accuracy": acc,
            "auc": auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
        }

    def cross_validate(self, X, y, cv=5, scoring="roc_auc"):
        print("Starting cross-validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True,
                              random_state=self.random_state)
        scores = []
        for i, (tr_idx, te_idx) in enumerate(
            tqdm(skf.split(X, y), total=cv, desc="CV folds"), 1
        ):
            print(f"Training fold {i}/{cv}...")
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            self.pipeline.fit(X_tr, y_tr)
            y_prob = self.pipeline.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            tqdm.write(f"Fold {i}/{cv} AUC: {auc:.4f}")
            scores.append(auc)
        self.cv_scores = np.array(scores)
        print("Cross-validation complete.")
        return self.cv_scores

    def get_feature_importance(self, X, y):
        print("Getting feature importances...")
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            return pd.DataFrame({
                "feature": X.columns,
                "importance": clf.feature_importances_
            })
        print("Using permutation importance...")
        perm = permutation_importance(
            self.pipeline, X, y, n_repeats=5,
            random_state=self.random_state
        )
        return pd.DataFrame({
            "feature": X.columns,
            "importance": perm.importances_mean
        })

    def get_model(self):
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        print("Returning trained model...")
        return self.pipeline.named_steps["classifier"]

    def get_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built.")
        print("Returning full pipeline...")
        return self.pipeline
