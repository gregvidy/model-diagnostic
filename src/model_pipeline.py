import pandas as pd
import numpy as np
import gc
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from tqdm import tqdm


# --- CategoryManager Class ---
class CategoryManager:
    """
    Manages the encoding of categorical features.
    Fits on training data to store categories and transforms new data
    based on the learned categories, mapping unseen categories to a missing code.
    """

    def __init__(self):
        self.category_map = {}  # Stores {column_name: pandas.CategoricalIndex}

    def fit(self, X: pd.DataFrame, categorical_columns: list):
        """
        Fits the CategoryManager by learning categories for specified columns.
        Adds a '__missing__' category and fills NaNs.
        """
        print("Fitting CategoryManager...")
        X_copy = X.copy()  # Work on a copy to avoid modifying original df during fit
        for col in categorical_columns:
            if col not in X_copy.columns:
                print(f"Warning: Column '{col}' not found in DataFrame for fitting.")
                continue
            # Ensure column is category type and handle NaNs consistently
            X_copy[col] = X_copy[col].astype("category")
            # Add '__missing__' explicitly before filling to ensure it's always available
            if "__missing__" not in X_copy[col].cat.categories:
                X_copy[col] = X_copy[col].cat.add_categories("__missing__")
            X_copy[col] = X_copy[col].fillna("__missing__")
            self.category_map[col] = X_copy[col].cat.categories
            print(
                f"  Fitted categories for column '{col}': {list(self.category_map[col])}"
            )
        print("CategoryManager fitting complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms categorical columns in the DataFrame to numerical codes
        based on the fitted categories. Unseen categories are mapped to the
        code for '__missing__'.
        """
        print("Transforming data using CategoryManager...")
        X_transformed = X.copy()
        for col, categories in self.category_map.items():
            if col not in X_transformed.columns:
                print(
                    f"Warning: Column '{col}' not found in DataFrame for transformation. Skipping."
                )
                continue

            # Convert to category type
            X_transformed[col] = X_transformed[col].astype("category")

            # Set categories to the ones learned during fit.
            # `set_categories` will make new categories NaN, and `fillna` will handle them.
            # Ensure '__missing__' is in categories for robustness
            if "__missing__" not in categories:
                # This should ideally not happen if fit() was called correctly
                # For safety, if it's missing, we'll try to add it.
                categories_list = list(categories)
                categories_list.append("__missing__")
                categories = pd.CategoricalIndex(categories_list)

            X_transformed[col] = X_transformed[col].cat.set_categories(categories)
            X_transformed[col] = X_transformed[col].fillna(
                "__missing__"
            )  # Fill with '__missing__' before getting codes

            # Convert categories to numerical codes
            X_transformed[col] = X_transformed[col].cat.codes
            print(f"  Transformed column '{col}' to numerical codes.")
        print("CategoryManager transformation complete.")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Fits the CategoryManager and then transforms the data.
        Convenience method for training phase.
        """
        self.fit(X, categorical_columns)
        return self.transform(X)

    def save(self, filepath: str):
        """Saves the learned category map to a file using pickle."""
        print(f"Saving category_map to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(self.category_map, f)
        print("Save complete.")

    def load(self, filepath: str):
        """Loads a category map from a file."""
        print(f"Loading category_map from {filepath}...")
        with open(filepath, "rb") as f:
            self.category_map = pickle.load(f)
        print("Load complete.")


# --- NumericalImputer Class ---
class NumericalImputer:
    """
    Imputes missing numerical values using a learned median.
    Fits on training data to store medians and transforms new data
    using those stored medians.
    """

    def __init__(self):
        self.imputation_values = {}  # Stores {column_name: median_value}

    def fit(self, X: pd.DataFrame, numerical_columns: list):
        """
        Fits the NumericalImputer by learning median values for specified columns.
        """
        print("Fitting NumericalImputer...")
        X_copy = X.copy()  # Work on a copy to avoid modifying original df during fit
        for col in numerical_columns:
            if col not in X_copy.columns:
                print(
                    f"Warning: Column '{col}' not found in DataFrame for fitting NumericalImputer. Skipping."
                )
                continue
            # Replace infinities with NaN first, then calculate median
            X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)
            median_val = X_copy[col].median()
            if pd.isna(median_val):
                # Fallback to 0 if all values are NaN in training data
                self.imputation_values[col] = 0
                print(
                    f"  Warning: All values in numerical column '{col}' are NaN. Imputing with 0."
                )
            else:
                self.imputation_values[col] = median_val
                print(f"  Fitted median for column '{col}': {median_val}")
        print("NumericalImputer fitting complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms numerical columns in the DataFrame by imputing missing values
        using the learned medians.
        """
        print("Transforming data using NumericalImputer...")
        X_transformed = X.copy()
        for col, median_val in self.imputation_values.items():
            if col not in X_transformed.columns:
                print(
                    f"Warning: Column '{col}' not found in DataFrame for transformation. Skipping."
                )
                continue
            # Replace infinities with NaN first, then fill NaNs
            X_transformed[col] = X_transformed[col].replace([np.inf, -np.inf], np.nan)
            X_transformed[col] = X_transformed[col].fillna(median_val)
            print(f"  Imputed column '{col}' with value: {median_val}")
        print("NumericalImputer transformation complete.")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
        """
        Fits the NumericalImputer and then transforms the data.
        Convenience method for training phase.
        """
        self.fit(X, numerical_columns)
        return self.transform(X)

    def save(self, filepath: str):
        """Saves the learned imputation values to a file using pickle."""
        print(f"Saving imputation_values to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(self.imputation_values, f)
        print("Save complete.")

    def load(self, filepath: str):
        """Loads imputation values from a file."""
        print(f"Loading imputation_values from {filepath}...")
        with open(filepath, "rb") as f:
            self.imputation_values = pickle.load(f)
        print("Load complete.")


# --- Core Preprocessing Functions ---
def preprocess_numerics_and_bools_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs core numeric and boolean preprocessing:
    - Downcasts float columns.
    - Converts 'Y'/'N' object columns and 'Is*' columns to 1/0.
    - Applies log10 transform to 'amount' related float columns.
    This function handles transformations that don't require learned parameters (like medians).
    """
    print("Starting core numeric and boolean preprocessing...")
    df_processed = df.copy()

    # Downcast float columns for memory efficiency
    float_cols = df_processed.select_dtypes(include=["float32", "float64"]).columns
    print(f"  Float columns to downcast: {list(float_cols)}")
    for col in float_cols:
        try:
            df_processed[col] = pd.to_numeric(df_processed[col], downcast="float")
            print(f"  Downcasted column: {col}")
        except Exception as e:
            print(f"  Error downcasting column {col}: {e}. Keeping as float64.")
            df_processed[col] = df_processed[col].astype("float64")

    # Handle 'Y'/'N' mappings for object columns and 'Is*' columns
    for col in df_processed.columns:
        if df_processed[col].dtype == "object":
            df_processed[col] = df_processed[col].replace(r"^\s*$", np.nan, regex=True)
            unique_vals = df_processed[col].dropna().unique()
            if set(unique_vals).issubset({"Y", "N"}):
                print(f"  Mapping Y/N to 1/0 in column: {col}")
                df_processed[col] = df_processed[col].map({"N": 0, "Y": 1})
        if col.startswith("Is") and df_processed[col].dtype in [
            "object",
            "int64",
            "bool",
        ]:
            if df_processed[col].dtype == "object" and set(
                df_processed[col].dropna().unique()
            ).issubset({"Y", "N"}):
                print(f"  Mapping Is* column Y/N to 1/0: {col}")
                df_processed[col] = df_processed[col].map({"N": 0, "Y": 1})
            elif df_processed[col].dtype in ["int64", "bool"]:
                try:
                    df_processed[col] = df_processed[col].astype(np.int8)
                    print(f"  Converted Is* column '{col}' to int8.")
                except Exception as e:
                    print(
                        f"  Could not convert Is* column '{col}' to int8: {e}. Keeping current dtype."
                    )

    # Apply log10 transform to 'amount'/'limit' related float columns
    current_float_cols = df_processed.select_dtypes(
        include=["float32", "float64"]
    ).columns
    for col in current_float_cols:
        if any(k in col.lower() for k in ["amt", "amount", "limit"]):
            print(f"  Applying log10 transform to column: {col}")
            vals = df_processed[col]
            df_processed[col] = np.log10(
                vals.fillna(0) + 1
            )  # Fill NaN with 0 before log transform
            print(f"  Log10 transformed column: {col}")

    print("Core numeric and boolean preprocessing complete.")
    return df_processed


# --- ModelPipeline Class ---
class ModelPipeline:
    def _init_lightgbm(self):
        print("Initializing LightGBM model...")
        return lgb.LGBMClassifier(
            n_estimators=1000,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
            objective="binary",  # For binary classification
            metric="auc",  # Evaluation metric
        )

    def __init__(self, model_type="xgboost", random_state=42):
        print(f"Initializing ModelPipeline with model type: {model_type}")
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._get_model(model_type)
        self.pipeline = None
        self.cv_scores = None
        self.category_manager = CategoryManager()  # Initialize CategoryManager
        self.numerical_imputer = NumericalImputer()  # Initialize NumericalImputer

        # Store column names after prepare_data for consistent handling across methods
        self.numerical_columns = []
        self.categorical_columns = []
        self.feature_columns = []  # To store the final list of feature names

    def _get_model(self, model_type):
        print(f"Retrieving model instance for type: {model_type}")
        models = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=1000,
                use_label_encoder=False,  # Deprecated in new XGBoost versions, good to set.
                eval_metric="logloss",  # Or 'auc' for binary classification
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
            ),
            "lightgbm": self._init_lightgbm(),
        }
        if model_type not in models:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return models[model_type]

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        exclude_columns: list = None,
        is_training: bool = True,
    ):
        """
        Preprocesses the input DataFrame, separates features (X) and target (y),
        identifies numerical and categorical columns, and applies learned transformations.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target variable column.
            exclude_columns (list, optional): List of columns to exclude from features X.
                                               Defaults to None.
            is_training (bool): If True, preprocessors (CategoryManager, NumericalImputer)
                                will be fitted. If False, only transformed.

        Returns:
            tuple: (X, y, numerical_columns, categorical_columns)
                   X and y are fully preprocessed and ready for model training/prediction.
        """
        print(f"Preparing data (is_training={is_training})...")

        if target_column not in df.columns:
            raise KeyError(f"Missing target column: {target_column} in raw DataFrame.")

        # Step 1: Separate target and features, and apply core transformations
        y = df[target_column].copy()
        columns_to_drop = [target_column] + (exclude_columns or [])
        X_raw = df.drop(columns=columns_to_drop, errors="ignore").copy()

        # Apply core preprocessing that doesn't require fitting
        X_processed_core = preprocess_numerics_and_bools_core(X_raw)

        # Identify numerical and categorical columns *after* core preprocessing
        # and before CategoryManager and NumericalImputer
        current_object_cols = X_processed_core.select_dtypes(
            include="object"
        ).columns.tolist()
        current_numeric_cols = X_processed_core.select_dtypes(
            include=["number"]
        ).columns.tolist()

        # Step 2: Handle categorical columns with CategoryManager
        if is_training:
            if not current_object_cols:
                print("No object columns found for CategoryManager to fit_transform.")
            else:
                X_transformed = self.category_manager.fit_transform(
                    X_processed_core, current_object_cols
                )
        else:  # is_training is False, so transform mode
            X_transformed = self.category_manager.transform(X_processed_core)

        # After category manager, some object columns are now numerical (int codes)
        # Update column lists. The categorical_columns will now refer to the columns
        # that *were* objects and are now numerical codes.
        # numerical_columns will refer to purely numerical columns (floats/ints)
        # that were not touched by category manager.
        self.categorical_columns = [
            col for col in current_object_cols if col in X_transformed.columns
        ]
        self.numerical_columns = [
            col for col in current_numeric_cols if col in X_transformed.columns
        ]

        print(f"Identified numerical columns for imputation: {self.numerical_columns}")
        print(f"Identified categorical columns (encoded): {self.categorical_columns}")

        # Step 3: Handle numerical columns with NumericalImputer
        if is_training:
            X_final = self.numerical_imputer.fit_transform(
                X_transformed, self.numerical_columns
            )
        else:
            X_final = self.numerical_imputer.transform(X_transformed)

        # Store the final feature column names for consistency
        self.feature_columns = X_final.columns.tolist()
        print(f"Final feature columns: {self.feature_columns}")

        gc.collect()  # Free up memory
        print("Data preparation complete.")
        return X_final, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3):
        """
        Splits data into training and testing sets using stratified sampling.
        """
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        print("Data splitting complete.")
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def split_data_by_date(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_column: str,
        split_date: str,
    ):
        """
        Splits data into training and testing sets based on a date column *before* full preprocessing.
        The data returned by this method will still need to be processed by `prepare_data`
        for each split.
        """
        print("Splitting data by date (pre-preprocessing)...")
        # Ensure date_column is datetime type for comparison
        if not pd.api.types.is_datetime64_any_dtype(X[date_column]):
            try:
                X[date_column] = pd.to_datetime(X[date_column])
                print(f"  Converted '{date_column}' to datetime type.")
            except Exception as e:
                raise TypeError(
                    f"Could not convert '{date_column}' to datetime format: {e}"
                )

        # Ensure split_date is datetime
        try:
            split_date = pd.to_datetime(split_date)
        except Exception as e:
            raise ValueError(
                f"Invalid split_date format: {e}. Please provide a valid date string."
            )

        X_train = X[X[date_column] < split_date].copy()
        X_test = X[X[date_column] >= split_date].copy()
        y_train = y.loc[X_train.index].copy()
        y_test = y.loc[X_test.index].copy()

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Return raw DFs for split, so prepare_data can be called on each
        print("Data splitting complete.")
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def build_pipeline(self):
        """
        Builds the scikit-learn pipeline for the model.
        The actual preprocessing steps (imputation, encoding) are handled by
        the ModelPipeline's `prepare_data` method directly, not within this sklearn.Pipeline.
        This sklearn.Pipeline primarily wraps the classifier.
        """
        print("Building pipeline...")

        # IdentityTransformer is used to allow the classifier to be the final step
        # The data fed to this pipeline is expected to be already fully preprocessed
        class IdentityTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        self.pipeline = Pipeline(
            [("identity", IdentityTransformer()), ("classifier", self.model)]
        )
        print("Pipeline built.")

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, show_training_log: bool = False
    ):
        """
        Trains the model using the preprocessed training data.
        X_train and y_train are expected to be the output of prepare_data(is_training=True).
        """
        print("Training model...")
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")

        # Ensure feature columns are consistent (important if X_train gets reindexed)
        if self.feature_columns and not X_train.columns.equals(
            pd.Index(self.feature_columns)
        ):
            print(
                "Warning: Training data columns do not match expected feature columns. Realigning."
            )
            X_train = X_train[self.feature_columns]

        if show_training_log and self.model_type == "xgboost":
            params = {
                "classifier__eval_set": [(X_train, y_train)],
                "classifier__verbose": True,
            }
            self.pipeline.fit(X_train, y_train, **params)
        elif show_training_log and self.model_type == "lightgbm":
            self.pipeline.fit(
                X_train, y_train, classifier__callbacks=[lgb.log_evaluation(period=50)]
            )
        else:
            self.pipeline.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluates the trained model on the test data.
        X_test and y_test are expected to be the output of prepare_data(is_training=False).
        """
        print("Evaluating model...")
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Ensure feature columns are consistent
        if self.feature_columns and not X_test.columns.equals(
            pd.Index(self.feature_columns)
        ):
            print(
                "Warning: Test data columns do not match expected feature columns. Realigning."
            )
            X_test = X_test[self.feature_columns]

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
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        return {
            "accuracy": acc,
            "auc": auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
        }

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new, unseen data.
        X_new is expected to be a raw DataFrame, which will be preprocessed internally.
        """
        print("Making predictions on new data...")
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Preprocess the new data using the *fitted* category_manager and numerical_imputer
        # from the training phase of the pipeline.
        X_new_preprocessed, _ = self.prepare_data(
            X_new.copy(),  # Pass a copy of the raw new data
            target_column=None,  # No target column in new data
            is_training=False,  # Crucial: only transform, do not fit
        )

        # Ensure feature columns are consistent
        if self.feature_columns and not X_new_preprocessed.columns.equals(
            pd.Index(self.feature_columns)
        ):
            print(
                "Warning: New data columns do not match expected feature columns. Realigning."
            )
            X_new_preprocessed = X_new_preprocessed[self.feature_columns]
            # Handle potentially missing columns in new data that were present in training
            missing_cols = set(self.feature_columns) - set(X_new_preprocessed.columns)
            for col in missing_cols:
                X_new_preprocessed[col] = 0  # Or a default value if appropriate
                print(
                    f"  Added missing feature column '{col}' to new data with default value 0."
                )

        predictions = self.pipeline.predict(X_new_preprocessed)
        print("Prediction complete.")
        return predictions

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Returns prediction probabilities for new, unseen data.
        X_new is expected to be a raw DataFrame, which will be preprocessed internally.
        """
        print("Calculating prediction probabilities on new data...")
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Preprocess the new data using the *fitted* category_manager and numerical_imputer
        X_new_preprocessed, _ = self.prepare_data(
            X_new.copy(), target_column=None, is_training=False
        )

        # Ensure feature columns are consistent
        if self.feature_columns and not X_new_preprocessed.columns.equals(
            pd.Index(self.feature_columns)
        ):
            print(
                "Warning: New data columns do not match expected feature columns. Realigning."
            )
            X_new_preprocessed = X_new_preprocessed[self.feature_columns]
            missing_cols = set(self.feature_columns) - set(X_new_preprocessed.columns)
            for col in missing_cols:
                X_new_preprocessed[col] = 0  # Or a default value if appropriate

        probabilities = self.pipeline.predict_proba(X_new_preprocessed)[:, 1]
        print("Probability calculation complete.")
        return probabilities

    def cross_validate(
        self,
        df_full: pd.DataFrame,
        target_column: str,
        exclude_columns: list = None,
        cv: int = 5,
    ):
        """
        Performs cross-validation on the data.
        A new CategoryManager and NumericalImputer are initialized for each fold to prevent data leakage.
        """
        print("Starting cross-validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = []

        # Store original preprocessors to restore after CV
        original_category_manager = self.category_manager
        original_numerical_imputer = self.numerical_imputer
        original_model = self.model

        # Separate X and y *once* from the full dataframe before splitting for CV
        # This X will contain original columns, not yet fully preprocessed by the main pipeline's managers
        y_full = df_full[target_column]
        columns_to_drop_cv = [target_column] + (exclude_columns or [])
        X_full = df_full.drop(columns=columns_to_drop_cv, errors="ignore")

        for i, (tr_idx, te_idx) in enumerate(
            tqdm(skf.split(X_full, y_full), total=cv, desc="CV folds"), 1
        ):
            print(f"--- Training fold {i}/{cv} ---")
            X_tr_raw, y_tr = X_full.iloc[tr_idx].copy(), y_full.iloc[tr_idx].copy()
            X_te_raw, y_te = X_full.iloc[te_idx].copy(), y_full.iloc[te_idx].copy()

            # Initialize NEW CategoryManager and NumericalImputer for each fold to avoid data leakage
            fold_category_manager = CategoryManager()
            fold_numerical_imputer = NumericalImputer()

            # Apply core preprocessing for this fold's training data
            X_tr_processed_core = preprocess_numerics_and_bools_core(X_tr_raw)

            # Identify numerical and categorical columns for this fold (after core preprocessing)
            fold_current_object_cols = X_tr_processed_core.select_dtypes(
                include="object"
            ).columns.tolist()
            fold_current_numeric_cols = X_tr_processed_core.select_dtypes(
                include=["number"]
            ).columns.tolist()

            # Fit and transform categorical data for this fold
            X_tr_categorical_transformed = fold_category_manager.fit_transform(
                X_tr_processed_core, fold_current_object_cols
            )

            # Fit and transform numerical data for this fold
            X_tr_final = fold_numerical_imputer.fit_transform(
                X_tr_categorical_transformed, fold_current_numeric_cols
            )

            # Apply same preprocessing to test data for this fold, using fitted managers
            X_te_processed_core = preprocess_numerics_and_bools_core(X_te_raw)
            X_te_categorical_transformed = fold_category_manager.transform(
                X_te_processed_core
            )
            X_te_final = fold_numerical_imputer.transform(X_te_categorical_transformed)

            # Ensure columns are consistent for the fold (if any reordering/missing in test set)
            fold_feature_columns = X_tr_final.columns.tolist()
            if not X_te_final.columns.equals(pd.Index(fold_feature_columns)):
                # Realign columns of test set to match training set, filling missing with 0
                missing_in_test = set(fold_feature_columns) - set(X_te_final.columns)
                for col in missing_in_test:
                    X_te_final[col] = 0  # Fill with 0 or a sensible default
                X_te_final = X_te_final[
                    fold_feature_columns
                ]  # Reorder and drop extra cols

            # Build a fresh pipeline for each fold to ensure independent training
            fold_model = self._get_model(self.model_type)  # Get a fresh model instance
            fold_pipeline = Pipeline(
                [
                    (
                        "identity",
                        IdentityTransformer(),
                    ),  # Use the nested IdentityTransformer
                    ("classifier", fold_model),
                ]
            )

            # Train the fold's pipeline
            fold_pipeline.fit(X_tr_final, y_tr)

            # Evaluate on the test set of the current fold
            y_prob = fold_pipeline.predict_proba(X_te_final)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            tqdm.write(f"Fold {i}/{cv} AUC: {auc:.4f}")
            scores.append(auc)

        self.cv_scores = np.array(scores)
        print(
            f"\nCross-validation complete. Mean AUC: {self.cv_scores.mean():.4f} +/- {self.cv_scores.std():.4f}"
        )
        # Restore the pipeline's main preprocessors and model after CV
        self.category_manager = original_category_manager
        self.numerical_imputer = original_numerical_imputer
        self.model = original_model
        self.build_pipeline()  # Rebuild the main pipeline with the original model
        return self.cv_scores

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Retrieves feature importances from the trained model.
        X and y are expected to be the fully preprocessed data used for training.
        """
        print("Getting feature importances...")
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            print("  Using native feature importances.")
            return pd.DataFrame(
                {"feature": X.columns, "importance": clf.feature_importances_}
            ).sort_values(by="importance", ascending=False)
        else:
            print(
                "  Using permutation importance (can be computationally intensive)..."
            )
            # For permutation importance, ensure the X passed is fully preprocessed
            # and matches the expected features after all transformations.
            perm = permutation_importance(
                self.pipeline,
                X,
                y,
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
            )
            return pd.DataFrame(
                {"feature": X.columns, "importance": perm.importances_mean}
            ).sort_values(by="importance", ascending=False)

    def get_model(self):
        """
        Returns the trained classifier model.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        print("Returning trained model...")
        return self.pipeline.named_steps["classifier"]

    def get_pipeline(self):
        """
        Returns the full scikit-learn pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built.")
        print("Returning full pipeline...")
        return self.pipeline

    def save_model(self, filepath: str):
        """Saves the trained ModelPipeline instance to a file."""
        print(f"Saving ModelPipeline to {filepath}...")
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built or trained. Cannot save.")

        # Saving the entire ModelPipeline instance will preserve the state of
        # self.category_manager, self.numerical_imputer, and self.model (within self.pipeline)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print("ModelPipeline saved successfully.")

    @classmethod
    def load_model(cls, filepath: str):
        """Loads a saved ModelPipeline instance from a file."""
        print(f"Loading ModelPipeline from {filepath}...")
        with open(filepath, "rb") as f:
            loaded_pipeline = pickle.load(f)
        print("ModelPipeline loaded successfully.")
        return loaded_pipeline
