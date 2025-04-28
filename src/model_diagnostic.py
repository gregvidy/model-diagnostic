import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings


class TestSuite:
    """
    A comprehensive toolkit for machine learning model diagnostics and evaluation.

    This class provides methods to analyze model performance, residuals, feature importance,
    and identifies potential issues through various slicing techniques.

    Parameters
    ----------
    data : pandas.DataFrame or Dict
        The dataset to use for diagnostics. If a DataFrame is provided, it will be
        automatically split into train and test sets. If a dictionary, it should contain
        'train', 'test', and optionally 'main' DataFrames.

    model : object
        The fitted model to evaluate. Should have predict() method, and predict_proba()
        for classification models.

    target_col : str, optional
        The name of the target column. If not provided, will attempt to infer from
        common target column names or use the last column.

    test_size : float, optional
        The proportion of the dataset to use for testing when splitting the data.
        Only used if data is provided as a DataFrame. Default is 0.2.

    random_state : int, optional
        Random seed for reproducible results. Default is 42.
    """

    def __init__(
        self, data=None, model=None, target_col=None, test_size=0.2, random_state=42
    ):
        """Initialize the TestSuite with data and model."""
        self.random_state = random_state

        # Store model if provided, or set to None for later
        self.model = model

        # Handle data input
        if data is not None:
            if isinstance(data, dict):
                self.data = data
                if "train" not in data or "test" not in data:
                    raise ValueError(
                        "Data dictionary must contain 'train' and 'test' keys"
                    )
            elif isinstance(data, pd.DataFrame):
                train_df, test_df = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
                self.data = {"main": data, "train": train_df, "test": test_df}
            else:
                raise ValueError(
                    "Data must be either a dictionary of DataFrames or a DataFrame"
                )

            # Set target column if provided, or try to infer
            self.target_col = target_col
            if self.target_col is None:
                self._infer_target_column()

            # Infer feature columns
            self._infer_feature_columns()

            # Determine if it's a classification or regression model
            if model is not None:
                self._determine_model_type()

    def set_data(self, data, pipeline=None, target_col=None, test_size=0.2):
        """
        Set or update the data used for diagnostics.

        If a raw DataFrame is provided, it will be split and fully preprocessed
        via the given pipeline before storage. If a dict is provided,
        it must contain 'X_train','X_test','y_train','y_test' and will be
        transformed via the pipeline's preprocessor first.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Raw DataFrame to transform or dict of existing splits.
        pipeline : ModelPipeline or sklearn Pipeline, optional
            Pipeline used to preprocess data. Required.
        target_col : str, optional
            Name of the target column. If not provided, inferred.
        test_size : float, default=0.2
            Fraction for test split when data is DataFrame.
        """
        import pandas as pd

        # Ensure pipeline provided
        if pipeline is None:
            raise ValueError('A pipeline must be provided for preprocessing')

        # Determine full pipeline
        full_pipe = pipeline.get_pipeline() if hasattr(pipeline, 'get_pipeline') else pipeline

        # Extract preprocessor step
        if hasattr(full_pipe, 'named_steps') and 'preprocessor' in full_pipe.named_steps:
            preproc = full_pipe.named_steps['preprocessor']
        else:
            raise ValueError(
                'Pipeline must expose a preprocessor step via named_steps["preprocessor"]'
            )

        # Case 1: existing splits dict
        if isinstance(data, dict):
            # Set target column
            if target_col:
                self.target_col = target_col
            elif not hasattr(self, 'target_col') or self.target_col is None:
                self.target_col = data['y_train'].name

            feat_names = preproc.get_feature_names_out()
            # Transform features
            X_train_p = preproc.transform(data['X_train'])
            X_test_p = preproc.transform(data['X_test'])

            # Rebuild DataFrames
            train_df = pd.DataFrame(
                X_train_p, columns=feat_names,
                index=data['X_train'].index
            )
            train_df[self.target_col] = data['y_train'].values

            test_df = pd.DataFrame(
                X_test_p, columns=feat_names,
                index=data['X_test'].index
            )
            test_df[self.target_col] = data['y_test'].values

            main_df = pd.concat([train_df, test_df], axis=0).sort_index()
            self.data = {'main': main_df, 'train': train_df, 'test': test_df}

        # Case 2: raw DataFrame
        elif isinstance(data, pd.DataFrame):
            # Set target column
            if target_col:
                self.target_col = target_col
            elif not hasattr(self, 'target_col') or self.target_col is None:
                self.target_col = self._infer_target_column(data)

            # Prepare raw X, y via custom pipeline if available
            if hasattr(pipeline, 'prepare_data'):
                X, y, num_cols, cat_cols = pipeline.prepare_data(
                    df=data, target_column=self.target_col
                )
                splits = pipeline.split_data(X, y, test_size=test_size)
            else:
                from sklearn.model_selection import train_test_split
                X = data.drop(columns=[self.target_col])
                y = data[self.target_col]
                train_all, test_all = train_test_split(
                    data, test_size=test_size, stratify=y, random_state=0
                )
                splits = {
                    'X_train': train_all.drop(columns=[self.target_col]),
                    'y_train': train_all[self.target_col],
                    'X_test': test_all.drop(columns=[self.target_col]),
                    'y_test': test_all[self.target_col]
                }

            feat_names = preproc.get_feature_names_out()
            X_train_p = preproc.transform(splits['X_train'])
            X_test_p = preproc.transform(splits['X_test'])

            train_df = pd.DataFrame(
                X_train_p, columns=feat_names,
                index=splits['X_train'].index
            )
            train_df[self.target_col] = splits['y_train'].values

            test_df = pd.DataFrame(
                X_test_p, columns=feat_names,
                index=splits['X_test'].index
            )
            test_df[self.target_col] = splits['y_test'].values

            main_df = pd.concat([train_df, test_df], axis=0).sort_index()
            self.data = {'main': main_df, 'train': train_df, 'test': test_df}

        else:
            raise ValueError('Data must be dict of splits or pandas DataFrame')

        # Infer and set metadata
        self._infer_target_column()
        self._infer_feature_columns()

        # Update model type if present
        if hasattr(self, 'model') and self.model is not None:
            self._determine_model_type()

    def set_model(self, model):
        """
        Set or update the model used for diagnostics.

        Parameters
        ----------
        model : object
            The fitted model to evaluate. Should have predict() method, and predict_proba()
            for classification models.
        """
        self.model = model

        # Determine model type if data is already set
        if hasattr(self, "data") and self.data is not None:
            self._determine_model_type()

    def _infer_target_column(self):
        """Infer target column from the data."""
        # Assume the target column is y or target if present
        train_df = self.data["train"]
        possible_targets = ["y", "target", "label", "class", "outcome", "response"]

        for col in possible_targets:
            if col in train_df.columns:
                self.target_col = col
                return

        # If no standard name is found, assume the last column is the target
        self.target_col = train_df.columns[-1]
        warnings.warn(
            f"Target column not identified. Using last column '{self.target_col}' as target."
        )

    def _infer_feature_columns(self):
        """Infer feature columns from the data."""
        train_df = self.data["train"]

        # Features are all columns except the target
        self.feature_cols = [col for col in train_df.columns if col != self.target_col]

    def _determine_model_type(self):
        """Determine if the model is for classification or regression."""
        # Check if model has predict_proba method (common in classifiers)
        self.is_classifier = hasattr(self.model, "predict_proba")

        # Additional check: look at target values
        target_values = self.data["train"][self.target_col].nunique()
        if target_values <= 10:  # Arbitrary threshold
            self.is_classifier = True

        # If we're still not sure, assume it's regression
        if not hasattr(self, "is_classifier"):
            self.is_classifier = False

    def _get_dataset(self, dataset: str = "test"):
        """Get the specified dataset partition."""
        if dataset not in self.data:
            raise ValueError(
                f"Dataset '{dataset}' not found. Available: {list(self.data.keys())}"
            )
        return self.data[dataset]

    def _get_model_name(self):
        """Get a string representation of the model type."""
        if self.model is None:
            return "Unknown"
        return type(self.model).__name__

    def _sample_data(
        self, df: pd.DataFrame, sample_size: int, random_state: int = None
    ):
        """Sample the DataFrame if it's larger than sample_size."""
        if random_state is None:
            random_state = self.random_state

        if len(df) > sample_size:
            return resample(
                df, n_samples=sample_size, random_state=random_state, replace=False
            )
        return df

    def _compute_residuals(self, df: pd.DataFrame):
        """
        Compute residuals for the given DataFrame.

        For classification: actual - predicted probability
        For regression: actual - predicted
        """
        self._validate_setup()

        X = df[self.feature_cols]
        y_true = df[self.target_col]

        if self.is_classifier:
            y_pred = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        else:
            y_pred = self.model.predict(X)

        return y_true - y_pred

    def _validate_setup(self):
        """Validate that data and model are properly set up."""
        if not hasattr(self, "data") or self.data is None:
            raise ValueError(
                "Data not set. Use set_data() to provide data before running diagnostics."
            )

        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not set. Use set_model() to provide a fitted model before running diagnostics."
            )

    def diagnose_residual_analysis(
        self,
        features: str = None,
        use_prediction: bool = False,
        dataset: str = "test",
        sample_size: int = 2000,
        random_state: int = None,
    ):
        """
        Analyze the relationship between model residuals and a specified feature.

        Creates a scatter plot showing the residuals (actual - predicted values) against
        a chosen feature or target variable. This can help identify patterns or
        heteroscedasticity in the model's predictions. For classification tasks,
        residuals are calculated using predicted probabilities of the positive class.

        Parameters
        ----------
        features : str, default=None
            The name of the feature to plot on the x-axis.
            Can be ignored when use_prediction is True.

        use_prediction : bool, default=False
            Whether to use the model prediction (predicted probability for classification)
            as x-axis.

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis.

        sample_size : int, default=2000
            Maximum number of points to plot. If the dataset is larger,
            a random subsample of this size will be used to improve visualization clarity.

        random_state : int, default=None
            Random seed for reproducible subsampling. If None, uses the TestSuite's random_state.

        Returns
        -------
        dict
            A dictionary containing the analysis results with the following keys:
            - plot: A matplotlib figure object with the residual plot
            - data: A pandas DataFrame with the analyzed data
            - statistics: Basic statistics about the residuals
        """
        self._validate_setup()

        # Input validation
        if not use_prediction and features is None:
            raise ValueError(
                "Either 'features' must be specified or 'use_prediction' must be True"
            )

        if features is not None and features not in self.feature_cols:
            raise ValueError(
                f"Feature '{features}' not found in data columns: {self.feature_cols}"
            )

        # Set random state if not provided
        if random_state is None:
            random_state = self.random_state

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Sample the dataset if needed
        df_sample = self._sample_data(df, sample_size, random_state)

        # Compute residuals
        residuals = self._compute_residuals(df_sample)

        # Prepare x-axis values
        if use_prediction:
            if self.is_classifier:
                x_values = self.model.predict_proba(df_sample[self.feature_cols])[:, 1]
                x_label = "Predicted Probability"
            else:
                x_values = self.model.predict(df_sample[self.feature_cols])
                x_label = "Predicted Value"
        else:
            x_values = df_sample[features].values
            x_label = features

        # Create result table
        result_df = pd.DataFrame({"x_value": x_values, "residual": residuals})

        # Calculate basic statistics
        stats = {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "min": np.min(residuals),
            "max": np.max(residuals),
            "abs_mean": np.mean(np.abs(residuals)),
        }

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, residuals, alpha=0.6, color="blue")
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"Residual Analysis: {x_label} vs. Residuals")

        # Add residual trend line if enough points
        if len(x_values) > 10:
            try:
                from scipy.stats import linregress

                slope, intercept, r_value, p_value, std_err = linregress(
                    x_values, residuals
                )
                x_line = np.array([min(x_values), max(x_values)])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, "r-", alpha=0.7)

                # Add trend information to stats
                stats["trend_slope"] = slope
                stats["trend_p_value"] = p_value
                stats["trend_r_squared"] = r_value**2
            except:
                pass

        plt.tight_layout()

        return {"plot": fig, "data": result_df, "statistics": stats}

    def diagnose_residual_interpret(
        self, dataset: str = "train", n_estimators: int = 100
    ):
        """
        Analyzes feature importance by examining their relationship with prediction residuals.

        This function trains a secondary model on the residuals of the primary model to understand
        which features might explain the errors in the primary model. Features with high importance
        in the secondary model suggest they have information the primary model isn't fully capturing.

        Parameters
        ----------
        dataset : {"main", "train", "test"}, default="train"
            Which dataset partition to use for the analysis.

        n_estimators : int, default=100
            Number of estimators to use in the random forest model for residual analysis.

        Returns
        -------
        dict
            A dictionary containing the analysis results with the following keys:
            - plot: A matplotlib figure object with the feature importance plot
            - importance: A pandas DataFrame with feature importance values
            - model: The trained residual model
        """
        self._validate_setup()

        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError("This function requires scikit-learn to be installed.")

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Compute residuals
        residuals = self._compute_residuals(df)

        # Train a model on the residuals
        X = df[self.feature_cols]
        residual_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=self.random_state
        )
        residual_model.fit(X, residuals)

        # Get feature importances
        importances = residual_model.feature_importances_

        # Create a DataFrame for the results
        features_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot horizontal bars for better readability with many features
        y_pos = np.arange(len(self.feature_cols))
        sorted_idx = np.argsort(importances)
        ax.barh(y_pos, importances[sorted_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.array(self.feature_cols)[sorted_idx])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel("Importance in Residual Model")
        ax.set_title("Feature Importance for Residual Model")

        plt.tight_layout()

        return {"plot": fig, "importance": features_df, "model": residual_model}

    def diagnose_residual_cluster(
        self,
        n_clusters: int = 3,
        features: List[str] = None,
        dataset: str = "test",
        sample_size: int = 2000,
        random_state: int = None,
    ):
        """
        Analyze model residuals by clustering data points and evaluating performance within clusters.

        This function identifies patterns in model residuals by grouping similar data points
        and analyzing how the model performs for each group. It helps detect subpopulations
        where the model might be consistently under- or over-predicting.

        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters to create

        features : List[str], default=None
            Features to use for clustering. If None, all features will be used.

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis.

        sample_size : int, default=2000
            Maximum number of points to use. If the dataset is larger,
            a random subsample of this size will be used.

        random_state : int, default=None
            Random seed for reproducible clustering and sampling.
            If None, uses the TestSuite's random_state.

        Returns
        -------
        dict
            A dictionary containing the analysis results with the following keys:
            - plot: A matplotlib figure object with visualizations of the clusters
            - clusters: A pandas DataFrame with the data points and their cluster assignments
            - stats: A pandas DataFrame with statistics for each cluster
        """
        self._validate_setup()

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("This function requires scikit-learn to be installed.")

        # Set random state if not provided
        if random_state is None:
            random_state = self.random_state

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Sample the dataset if needed
        df_sample = self._sample_data(df, sample_size, random_state)

        # Select features for clustering
        if features is None:
            features = self.feature_cols
        else:
            # Validate that all requested features exist
            missing_features = [f for f in features if f not in self.feature_cols]
            if missing_features:
                raise ValueError(f"Features not found in data: {missing_features}")

        # Standardize features for clustering
        X = df_sample[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)

        # Compute residuals and add cluster information
        residuals = self._compute_residuals(df_sample)
        cluster_df = pd.DataFrame({"cluster": clusters, "residual": residuals})

        # For each feature, add to the result DataFrame
        for feature in features:
            cluster_df[feature] = df_sample[feature].values

        # Add predicted and actual values
        if self.is_classifier:
            cluster_df["actual"] = df_sample[self.target_col]
            cluster_df["predicted_prob"] = self.model.predict_proba(
                df_sample[self.feature_cols]
            )[:, 1]
        else:
            cluster_df["actual"] = df_sample[self.target_col]
            cluster_df["predicted"] = self.model.predict(df_sample[self.feature_cols])

        # Compute statistics by cluster
        stats = []
        for i in range(n_clusters):
            cluster_data = cluster_df[cluster_df["cluster"] == i]

            cluster_stat = {
                "cluster": i,
                "size": len(cluster_data),
                "mean_residual": cluster_data["residual"].mean(),
                "abs_mean_residual": cluster_data["residual"].abs().mean(),
                "std_residual": cluster_data["residual"].std(),
                "min_residual": cluster_data["residual"].min(),
                "max_residual": cluster_data["residual"].max(),
            }

            # Add feature means for this cluster
            for feature in features:
                cluster_stat[f"mean_{feature}"] = cluster_data[feature].mean()

            stats.append(cluster_stat)

        stats_df = pd.DataFrame(stats)

        # Create visualization
        try:
            # Try to create a more advanced visualization with seaborn
            import seaborn as sns

            has_seaborn = True
        except ImportError:
            has_seaborn = False

        if has_seaborn:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Violin plot of residuals by cluster
            sns.violinplot(x="cluster", y="residual", data=cluster_df, ax=ax1)
            ax1.set_title("Residual Distribution by Cluster")
            ax1.set_xlabel("Cluster")
            ax1.set_ylabel("Residual")
            ax1.axhline(y=0, color="r", linestyle="--")

            # Bar plot of cluster sizes with mean residuals
            sizes = stats_df["size"].values
            ax2.bar(stats_df["cluster"], sizes, alpha=0.7)
            ax2.set_title("Cluster Sizes and Mean Residuals")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Count")

            # Add text with mean residual
            for i, (count, mean) in enumerate(
                zip(sizes, stats_df["mean_residual"].values)
            ):
                ax2.text(i, count + max(sizes) * 0.05, f"Mean: {mean:.3f}", ha="center")
        else:
            # Basic visualization
            fig, ax = plt.subplots(figsize=(10, 6))

            # Box plot of residuals by cluster
            cluster_groups = [
                cluster_df[cluster_df["cluster"] == i]["residual"]
                for i in range(n_clusters)
            ]
            ax.boxplot(cluster_groups)
            ax.set_title("Residual Distribution by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Residual")
            ax.axhline(y=0, color="r", linestyle="--")

        plt.tight_layout()

        return {
            "plot": fig,
            "clusters": cluster_df,
            "stats": stats_df,
            "kmeans": kmeans,
        }

    def diagnose_slicing_accuracy(
        self,
        slice_feature: str,
        n_bins: int = 10,
        dataset: str = "test",
        min_samples: int = 50,
    ):
        """
        Identify low-accuracy regions based on specified slicing features.

        This function divides the data into bins based on a specified feature and evaluates
        the model's performance within each bin. It helps identify regions where the model
        performs poorly, which can guide targeted improvements.

        Parameters
        ----------
        slice_feature : str
            Feature to use for slicing the data

        n_bins : int, default=10
            Number of bins to create for numerical features

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis

        min_samples : int, default=50
            Minimum number of samples required in a bin for it to be included in results

        Returns
        -------
        dict
            A dictionary containing the analysis results with the following keys:
            - plot: A matplotlib figure object showing performance across slices
            - slices: A pandas DataFrame with performance metrics for each slice
            - worst_slice: The slice with the worst performance
            - best_slice: The slice with the best performance
        """
        self._validate_setup()

        # Validate slice feature
        if slice_feature not in self.feature_cols:
            raise ValueError(
                f"Slice feature '{slice_feature}' not found in data columns"
            )

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Create slices based on feature type
        feature_values = df[slice_feature]

        if pd.api.types.is_numeric_dtype(feature_values):
            # For numeric features, create equal-width bins
            bins = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
            labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
            df["slice"] = pd.cut(
                feature_values, bins=bins, labels=labels, include_lowest=True
            )
        else:
            # For categorical features, use categories as is
            df["slice"] = feature_values

        # Compute predictions
        X = df[self.feature_cols]
        y_true = df[self.target_col]

        if self.is_classifier:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            y_pred = self.model.predict(X)

            # Compute metrics for each slice
            slice_metrics = []
            unique_slices = df["slice"].dropna().unique()

            for slice_val in unique_slices:
                idx = df["slice"] == slice_val

                # Skip slices with too few samples
                if sum(idx) < min_samples:
                    continue

                metrics = {
                    "slice_value": str(slice_val),
                    "count": sum(idx),
                    "accuracy": accuracy_score(y_true[idx], y_pred[idx]),
                    "precision": precision_score(
                        y_true[idx], y_pred[idx], zero_division=0
                    ),
                    "recall": recall_score(y_true[idx], y_pred[idx], zero_division=0),
                    "f1": f1_score(y_true[idx], y_pred[idx], zero_division=0),
                }
                slice_metrics.append(metrics)

            # Performance metric to plot
            performance_metric = "accuracy"
            metric_name = "Accuracy"
            higher_is_better = True
        else:
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            y_pred = self.model.predict(X)

            # Compute metrics for each slice
            slice_metrics = []
            unique_slices = df["slice"].dropna().unique()

            for slice_val in unique_slices:
                idx = df["slice"] == slice_val

                # Skip slices with too few samples
                if sum(idx) < min_samples:
                    continue

                metrics = {
                    "slice_value": str(slice_val),
                    "count": sum(idx),
                    "mse": mean_squared_error(y_true[idx], y_pred[idx]),
                    "mae": mean_absolute_error(y_true[idx], y_pred[idx]),
                    "r2": r2_score(y_true[idx], y_pred[idx]),
                }
                slice_metrics.append(metrics)

            # Performance metric to plot
            performance_metric = "mse"
            metric_name = "Mean Squared Error"
            higher_is_better = False

        # Convert to DataFrame
        metrics_df = pd.DataFrame(slice_metrics)

        # Sort by performance metric
        if higher_is_better:
            metrics_df = metrics_df.sort_values(performance_metric)
        else:
            metrics_df = metrics_df.sort_values(performance_metric, ascending=False)

        # Find worst and best slices
        if not metrics_df.empty:
            worst_slice = (
                metrics_df.iloc[0].to_dict()
                if not higher_is_better
                else metrics_df.iloc[-1].to_dict()
            )
            best_slice = (
                metrics_df.iloc[-1].to_dict()
                if not higher_is_better
                else metrics_df.iloc[0].to_dict()
            )
        else:
            worst_slice = {}
            best_slice = {}

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        if not metrics_df.empty:
            # Sort by slice value for visualization (if numeric)
            if pd.api.types.is_numeric_dtype(feature_values):
                try:
                    # Try to extract lower bound of the range and sort
                    metrics_df["sort_val"] = metrics_df["slice_value"].apply(
                        lambda x: float(x.split("-")[0])
                    )
                    metrics_df = metrics_df.sort_values("sort_val")
                except:
                    pass

            # Plot
            x = metrics_df["slice_value"]
            y = metrics_df[performance_metric]

            bars = ax.bar(x, y, color="skyblue")

            # Add count labels on top of bars
            for i, (bar, count) in enumerate(zip(bars, metrics_df["count"])):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(y) - min(y)) * 0.02,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            # Highlight worst performing slice
            if higher_is_better:
                worst_idx = np.argmin(y)
            else:
                worst_idx = np.argmax(y)

            bars[worst_idx].set_color("salmon")

            ax.set_xlabel(f"{slice_feature} Slice")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} by {slice_feature} Slice")
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
        else:
            ax.text(
                0.5,
                0.5,
                "Not enough data to create slices",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        return {
            "plot": fig,
            "slices": metrics_df,
            "worst_slice": worst_slice,
            "best_slice": best_slice,
        }

    def diagnose_slicing_overfit(
        self,
        slice_feature1: str,
        slice_feature2: str = None,
        n_bins: int = 5,
        dataset: str = "main",
    ):
        """
        Identify overfit regions based on one or two slicing features.

        This function divides the data based on one or two features and measures
        the difference in performance between training and test datasets for each slice.
        Large differences indicate potential overfit regions.

        Parameters
        ----------
        slice_feature1 : str
            First feature to use for slicing

        slice_feature2 : str, default=None
            Optional second feature for 2D slicing

        n_bins : int, default=5
            Number of bins to create for each numerical feature

        dataset : {"main"}, default="main"
            Must be "main" as we need both train and test data

        Returns
        -------
        dict
            A dictionary containing the analysis results with the following keys:
            - plot: A matplotlib figure object showing the performance gap across slices
            - slices: A pandas DataFrame with train/test performance for each slice
            - worst_overfit: The slice with the largest train/test performance gap
        """
        self._validate_setup()

        if dataset != "main":
            warnings.warn(
                "This function requires the full dataset. Using 'main' dataset."
            )
            dataset = "main"

        # Validate slice features
        for feature in [slice_feature1, slice_feature2]:
            if feature is not None and feature not in self.feature_cols:
                raise ValueError(f"Slice feature '{feature}' not found in data columns")

        # Get the full dataset
        df = self._get_dataset(dataset)

        # Split into train and test based on whether the data point is in the training set
        train_indices = df.index.isin(self.data["train"].index)
        test_indices = df.index.isin(self.data["test"].index)

        # Create slices based on feature type for the first feature
        if pd.api.types.is_numeric_dtype(df[slice_feature1]):
            bins1 = np.linspace(
                df[slice_feature1].min(), df[slice_feature1].max(), n_bins + 1
            )
            labels1 = [f"{bins1[i]:.2f}-{bins1[i+1]:.2f}" for i in range(n_bins)]
            df["slice1"] = pd.cut(
                df[slice_feature1], bins=bins1, labels=labels1, include_lowest=True
            )
        else:
            df["slice1"] = df[slice_feature1]

        # Create slices for second feature if provided
        if slice_feature2 is not None:
            if pd.api.types.is_numeric_dtype(df[slice_feature2]):
                bins2 = np.linspace(
                    df[slice_feature2].min(), df[slice_feature2].max(), n_bins + 1
                )
                labels2 = [f"{bins2[i]:.2f}-{bins2[i+1]:.2f}" for i in range(n_bins)]
                df["slice2"] = pd.cut(
                    df[slice_feature2], bins=bins2, labels=labels2, include_lowest=True
                )
            else:
                df["slice2"] = df[slice_feature2]

            # Create combined slice label
            df["slice"] = df["slice1"].astype(str) + " & " + df["slice2"].astype(str)
        else:
            df["slice"] = df["slice1"]

        # Compute predictions
        X = df[self.feature_cols]
        y_true = df[self.target_col]

        if self.is_classifier:
            from sklearn.metrics import accuracy_score

            y_pred = self.model.predict(X)

            # Compute metrics for each slice
            slice_metrics = []
            unique_slices = df["slice"].dropna().unique()

            for slice_val in unique_slices:
                idx = df["slice"] == slice_val

                # Get train and test indices for this slice
                train_slice = idx & train_indices
                test_slice = idx & test_indices

                # Skip slices with too few samples
                if sum(train_slice) < 10 or sum(test_slice) < 10:
                    continue

                # Calculate metrics
                train_acc = accuracy_score(y_true[train_slice], y_pred[train_slice])
                test_acc = accuracy_score(y_true[test_slice], y_pred[test_slice])

                metrics = {
                    "slice_value": str(slice_val),
                    "train_count": sum(train_slice),
                    "test_count": sum(test_slice),
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "accuracy_diff": train_acc - test_acc,
                }
                slice_metrics.append(metrics)

            # Performance metric
            performance_diff = "accuracy_diff"
            metric_name = "Accuracy"
        else:
            from sklearn.metrics import mean_squared_error

            y_pred = self.model.predict(X)

            # Compute metrics for each slice
            slice_metrics = []
            unique_slices = df["slice"].dropna().unique()

            for slice_val in unique_slices:
                idx = df["slice"] == slice_val

                # Get train and test indices for this slice
                train_slice = idx & train_indices
                test_slice = idx & test_indices

                # Skip slices with too few samples
                if sum(train_slice) < 10 or sum(test_slice) < 10:
                    continue

                # Calculate metrics
                train_mse = mean_squared_error(y_true[train_slice], y_pred[train_slice])
                test_mse = mean_squared_error(y_true[test_slice], y_pred[test_slice])

                metrics = {
                    "slice_value": str(slice_val),
                    "train_count": sum(train_slice),
                    "test_count": sum(test_slice),
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "mse_ratio": (
                        test_mse / train_mse if train_mse > 0 else float("inf")
                    ),
                }
                slice_metrics.append(metrics)

            # Performance metric
            performance_diff = "mse_ratio"
            metric_name = "MSE"

        # Convert to DataFrame
        metrics_df = pd.DataFrame(slice_metrics)

        # Sort by the difference/ratio to identify problematic slices
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values(performance_diff, ascending=False)
            worst_overfit = metrics_df.iloc[0].to_dict()
        else:
            worst_overfit = {}

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        if not metrics_df.empty:
            n_slices = len(metrics_df)
            x = np.arange(n_slices)

            if self.is_classifier:
                train_perf = metrics_df["train_accuracy"]
                test_perf = metrics_df["test_accuracy"]

                ax.bar(
                    x - 0.2,
                    train_perf,
                    width=0.4,
                    label="Train Accuracy",
                    color="blue",
                    alpha=0.7,
                )
                ax.bar(
                    x + 0.2,
                    test_perf,
                    width=0.4,
                    label="Test Accuracy",
                    color="orange",
                    alpha=0.7,
                )

                # Secondary axis for difference
                ax2 = ax.twinx()
                ax2.plot(
                    x,
                    metrics_df["accuracy_diff"],
                    "r-",
                    linewidth=2,
                    label="Difference",
                )
                ax2.set_ylabel("Accuracy Difference (Train - Test)")

            else:
                train_perf = metrics_df["train_mse"]
                test_perf = metrics_df["test_mse"]

                ax.bar(
                    x - 0.2,
                    train_perf,
                    width=0.4,
                    label="Train MSE",
                    color="blue",
                    alpha=0.7,
                )
                ax.bar(
                    x + 0.2,
                    test_perf,
                    width=0.4,
                    label="Test MSE",
                    color="orange",
                    alpha=0.7,
                )

                # Secondary axis for ratio
                ax2 = ax.twinx()
                ax2.plot(x, metrics_df["mse_ratio"], "r-", linewidth=2, label="Ratio")
                ax2.set_ylabel("MSE Ratio (Test / Train)")

            # Labels and legends
            ax.set_xlabel("Data Slice")
            ax.set_ylabel(f"{metric_name}")
            ax.set_title(f"Train vs. Test {metric_name} by Data Slice")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df["slice_value"], rotation=45, ha="right")

            # Create a combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.tight_layout()
        else:
            ax.text(
                0.5,
                0.5,
                "Not enough data to create slices with sufficient samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        return {"plot": fig, "slices": metrics_df, "worst_overfit": worst_overfit}

    def diagnose_fairness(
        self,
        protected_features: List[str],
        reference_groups: Dict[str, Any] = None,
        dataset: str = "test",
    ):
        """
        Evaluate fairness metrics across different protected and reference groups.

        This function computes various fairness metrics to evaluate model bias across different
        demographic or protected attribute groups. It helps identify potential discriminatory
        patterns in model predictions.

        Parameters
        ----------
        protected_features : List[str]
            Features to use for fairness evaluation (e.g., gender, race)

        reference_groups : Dict[str, Any], default=None
            Dictionary mapping feature names to their reference values.
            If None, the most common value for each feature will be used as reference.

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis

        Returns
        -------
        dict
            A dictionary containing the fairness analysis results with the following keys:
            - metrics: A pandas DataFrame with fairness metrics for each group
            - summary: A pandas DataFrame with summary statistics
            - plot: A matplotlib figure object showing the fairness metrics
        """
        self._validate_setup()

        if not self.is_classifier:
            warnings.warn(
                "Fairness diagnostics are primarily designed for classification models."
            )

        # Validate protected features
        for feature in protected_features:
            if feature not in self.feature_cols:
                raise ValueError(
                    f"Protected feature '{feature}' not found in data columns"
                )

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Create reference groups dictionary if not provided
        if reference_groups is None:
            reference_groups = {}
            for feature in protected_features:
                # Use most common value as reference
                most_common = df[feature].value_counts().index[0]
                reference_groups[feature] = most_common

        # Compute predictions
        X = df[self.feature_cols]
        y_true = df[self.target_col]

        if self.is_classifier:
            y_pred = self.model.predict(X)
            y_proba = None
            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
            y_proba = None

        # Compute metrics for each protected feature
        fairness_metrics = []

        for feature in protected_features:
            ref_value = reference_groups.get(feature)

            # Get unique values for this feature
            unique_values = df[feature].unique()

            for value in unique_values:
                if value == ref_value:
                    is_reference = True
                else:
                    is_reference = False

                # Filter data for this group
                group_idx = df[feature] == value
                group_size = sum(group_idx)

                if group_size < 10:  # Skip very small groups
                    continue

                # Compute metrics for this group
                group_metrics = {
                    "feature": feature,
                    "group_value": value,
                    "is_reference": is_reference,
                    "group_size": group_size,
                    "group_proportion": group_size / len(df),
                }

                # Performance metrics
                if self.is_classifier:
                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score,
                        roc_auc_score,
                    )

                    group_metrics["accuracy"] = accuracy_score(
                        y_true[group_idx], y_pred[group_idx]
                    )

                    group_metrics["precision"] = precision_score(
                        y_true[group_idx], y_pred[group_idx], zero_division=0
                    )

                    group_metrics["recall"] = recall_score(
                        y_true[group_idx], y_pred[group_idx], zero_division=0
                    )

                    group_metrics["f1"] = f1_score(
                        y_true[group_idx], y_pred[group_idx], zero_division=0
                    )

                    if y_proba is not None:
                        try:
                            group_metrics["auc"] = roc_auc_score(
                                y_true[group_idx], y_proba[group_idx]
                            )
                        except:
                            group_metrics["auc"] = np.nan

                    # Positive prediction rate
                    group_metrics["positive_rate"] = np.mean(y_pred[group_idx] == 1)

                    # True positive rate (TPR) / Recall
                    pos_idx = y_true[group_idx] == 1
                    if sum(pos_idx) > 0:
                        group_metrics["tpr"] = np.mean(y_pred[group_idx][pos_idx] == 1)
                    else:
                        group_metrics["tpr"] = np.nan

                    # False positive rate (FPR)
                    neg_idx = y_true[group_idx] == 0
                    if sum(neg_idx) > 0:
                        group_metrics["fpr"] = np.mean(y_pred[group_idx][neg_idx] == 1)
                    else:
                        group_metrics["fpr"] = np.nan

                else:
                    from sklearn.metrics import (
                        mean_squared_error,
                        mean_absolute_error,
                        r2_score,
                    )

                    group_metrics["mse"] = mean_squared_error(
                        y_true[group_idx], y_pred[group_idx]
                    )

                    group_metrics["mae"] = mean_absolute_error(
                        y_true[group_idx], y_pred[group_idx]
                    )

                    group_metrics["r2"] = r2_score(y_true[group_idx], y_pred[group_idx])

                fairness_metrics.append(group_metrics)

        # Convert to DataFrame
        metrics_df = pd.DataFrame(fairness_metrics)

        # Calculate fairness metrics comparing each group to reference
        fairness_summary = []

        for feature in protected_features:
            ref_value = reference_groups.get(feature)
            feature_metrics = metrics_df[metrics_df["feature"] == feature]

            # Get reference group metrics
            ref_metrics = feature_metrics[feature_metrics["group_value"] == ref_value]
            if len(ref_metrics) == 0:
                continue

            ref_row = ref_metrics.iloc[0]

            # Compare each non-reference group to reference
            for _, row in feature_metrics[
                feature_metrics["group_value"] != ref_value
            ].iterrows():
                comparison = {
                    "feature": feature,
                    "reference_value": ref_value,
                    "comparison_value": row["group_value"],
                    "ref_size": ref_row["group_size"],
                    "comparison_size": row["group_size"],
                }

                # Calculate disparities
                if self.is_classifier:
                    # Demographic parity ratio (positive prediction rate ratio)
                    if ref_row["positive_rate"] > 0:
                        comparison["demographic_parity_ratio"] = (
                            row["positive_rate"] / ref_row["positive_rate"]
                        )
                    else:
                        comparison["demographic_parity_ratio"] = np.nan

                    # Equal opportunity ratio (TPR ratio)
                    if ref_row["tpr"] > 0:
                        comparison["equal_opportunity_ratio"] = (
                            row["tpr"] / ref_row["tpr"]
                        )
                    else:
                        comparison["equal_opportunity_ratio"] = np.nan

                    # Equalized odds ratios (TPR and FPR ratios)
                    if ref_row["fpr"] > 0:
                        comparison["equalized_odds_fpr_ratio"] = (
                            row["fpr"] / ref_row["fpr"]
                        )
                    else:
                        comparison["equalized_odds_fpr_ratio"] = np.nan

                    # Performance disparities
                    comparison["accuracy_ratio"] = (
                        row["accuracy"] / ref_row["accuracy"]
                        if ref_row["accuracy"] > 0
                        else np.nan
                    )
                    comparison["precision_ratio"] = (
                        row["precision"] / ref_row["precision"]
                        if ref_row["precision"] > 0
                        else np.nan
                    )
                    comparison["recall_ratio"] = (
                        row["recall"] / ref_row["recall"]
                        if ref_row["recall"] > 0
                        else np.nan
                    )
                    comparison["f1_ratio"] = (
                        row["f1"] / ref_row["f1"] if ref_row["f1"] > 0 else np.nan
                    )

                    if "auc" in row and "auc" in ref_row:
                        comparison["auc_ratio"] = (
                            row["auc"] / ref_row["auc"]
                            if ref_row["auc"] > 0
                            else np.nan
                        )
                else:
                    # Performance ratios for regression
                    if ref_row["mse"] > 0:
                        comparison["mse_ratio"] = row["mse"] / ref_row["mse"]
                    else:
                        comparison["mse_ratio"] = np.nan

                    if ref_row["mae"] > 0:
                        comparison["mae_ratio"] = row["mae"] / ref_row["mae"]
                    else:
                        comparison["mae_ratio"] = np.nan

                    # R² can be negative, so ratio might not be meaningful
                    if ref_row["r2"] > 0 and row["r2"] > 0:
                        comparison["r2_ratio"] = row["r2"] / ref_row["r2"]
                    else:
                        comparison["r2_ratio"] = np.nan

                fairness_summary.append(comparison)

        # Convert to DataFrame
        summary_df = pd.DataFrame(fairness_summary)

        # Create visualization
        fig = None

        if not metrics_df.empty and self.is_classifier:
            fig, axes = plt.subplots(
                len(protected_features), 3, figsize=(15, 5 * len(protected_features))
            )

            # If only one protected feature, make axes a 2D array for consistent indexing
            if len(protected_features) == 1:
                axes = np.array([axes])

            for i, feature in enumerate(protected_features):
                feature_metrics = metrics_df[metrics_df["feature"] == feature]

                if len(feature_metrics) == 0:
                    continue

                # Get unique values for this feature and their counts
                values = feature_metrics["group_value"].values
                counts = feature_metrics["group_size"].values

                # Sort by group size
                sort_idx = np.argsort(counts)[::-1]
                values = values[sort_idx]
                counts = counts[sort_idx]

                # Plot group sizes
                axes[i, 0].bar(values, counts)
                axes[i, 0].set_title(f"Group Sizes for {feature}")
                axes[i, 0].set_xlabel(feature)
                axes[i, 0].set_ylabel("Count")

                # Plot positive prediction rates
                pos_rates = feature_metrics["positive_rate"].values[sort_idx]
                axes[i, 1].bar(values, pos_rates)
                axes[i, 1].set_title(f"Positive Prediction Rates for {feature}")
                axes[i, 1].set_xlabel(feature)
                axes[i, 1].set_ylabel("Positive Rate")

                # Plot performance metrics
                metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
                metric_values = {
                    m: feature_metrics[m].values[sort_idx] for m in metrics_to_plot
                }

                for j, metric in enumerate(metrics_to_plot):
                    axes[i, 2].bar(
                        values + j * 0.2 - 0.3,
                        metric_values[metric],
                        width=0.2,
                        label=metric,
                    )

                axes[i, 2].set_title(f"Performance Metrics for {feature}")
                axes[i, 2].set_xlabel(feature)
                axes[i, 2].set_ylabel("Metric Value")
                axes[i, 2].legend()

            plt.tight_layout()

        return {"metrics": metrics_df, "summary": summary_df, "plot": fig}

    def diagnose_robustness(
        self,
        perturbation_scales: List[float] = [0.1, 0.2, 0.5, 1.0],
        n_samples: int = 100,
        dataset: str = "test",
        feature_subset: List[str] = None,
    ):
        """
        Evaluate model robustness by measuring performance under feature perturbations.

        This function tests the model's stability by adding random noise to features
        and measuring how much the predictions change. It helps identify features
        where small changes can cause large shifts in predictions.

        Parameters
        ----------
        perturbation_scales : List[float], default=[0.1, 0.2, 0.5, 1.0]
            Standard deviation scales for the perturbations as fractions of the feature std

        n_samples : int, default=100
            Number of samples to use for robustness testing

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis

        feature_subset : List[str], default=None
            Subset of features to perturb. If None, all numerical features will be used.

        Returns
        -------
        dict
            A dictionary containing the robustness analysis results with the following keys:
            - plot: A matplotlib figure showing performance under different perturbation levels
            - perturbations: A pandas DataFrame with detailed results
            - feature_sensitivity: A pandas DataFrame ranking features by sensitivity
        """
        self._validate_setup()

        # Get the specified dataset
        df = self._get_dataset(dataset)

        # Sample data if needed
        df_sample = self._sample_data(df, n_samples)

        # Select features to perturb (only numerical features)
        if feature_subset is None:
            perturb_features = [
                f
                for f in self.feature_cols
                if pd.api.types.is_numeric_dtype(df_sample[f])
            ]
        else:
            # Validate feature subset
            missing_features = [f for f in feature_subset if f not in self.feature_cols]
            if missing_features:
                raise ValueError(f"Features not found: {missing_features}")

            # Only keep numerical features
            perturb_features = [
                f for f in feature_subset if pd.api.types.is_numeric_dtype(df_sample[f])
            ]

            if not perturb_features:
                raise ValueError(
                    "No numerical features found in the provided feature subset"
                )

        # Get original predictions
        X_orig = df_sample[self.feature_cols]
        y_true = df_sample[self.target_col]

        if self.is_classifier:
            y_orig_pred = self.model.predict(X_orig)
            y_orig_proba = None
            if hasattr(self.model, "predict_proba"):
                y_orig_proba = self.model.predict_proba(X_orig)[:, 1]
        else:
            y_orig_pred = self.model.predict(X_orig)

        # Compute feature standard deviations for scaling perturbations
        feature_stds = df_sample[perturb_features].std()

        # Results for all perturbations
        all_results = []

        # For each perturbation scale
        for scale in perturbation_scales:
            # For each feature to perturb
            for feature in perturb_features:
                # Create a copy of original data
                X_perturb = X_orig.copy()

                # Add noise to the feature
                noise = np.random.normal(
                    0, scale * feature_stds[feature], size=len(X_perturb)
                )
                X_perturb[feature] = X_perturb[feature] + noise

                # Get new predictions
                if self.is_classifier:
                    y_new_pred = self.model.predict(X_perturb)
                    y_new_proba = None
                    if hasattr(self.model, "predict_proba"):
                        y_new_proba = self.model.predict_proba(X_perturb)[:, 1]
                else:
                    y_new_pred = self.model.predict(X_perturb)

                # Calculate metrics
                result = {
                    "feature": feature,
                    "perturbation_scale": scale,
                    "prediction_changes": np.mean(y_new_pred != y_orig_pred),
                }

                if self.is_classifier:
                    from sklearn.metrics import accuracy_score

                    result["original_accuracy"] = accuracy_score(y_true, y_orig_pred)
                    result["perturbed_accuracy"] = accuracy_score(y_true, y_new_pred)
                    result["accuracy_change"] = (
                        result["original_accuracy"] - result["perturbed_accuracy"]
                    )

                    if y_orig_proba is not None and y_new_proba is not None:
                        result["probability_change_mean"] = np.mean(
                            np.abs(y_new_proba - y_orig_proba)
                        )
                        result["probability_change_max"] = np.max(
                            np.abs(y_new_proba - y_orig_proba)
                        )
                else:
                    from sklearn.metrics import mean_squared_error

                    result["original_mse"] = mean_squared_error(y_true, y_orig_pred)
                    result["perturbed_mse"] = mean_squared_error(y_true, y_new_pred)
                    result["mse_change"] = (
                        result["perturbed_mse"] - result["original_mse"]
                    )
                    result["prediction_change_mean"] = np.mean(
                        np.abs(y_new_pred - y_orig_pred)
                    )
                    result["prediction_change_max"] = np.max(
                        np.abs(y_new_pred - y_orig_pred)
                    )

                all_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        # Calculate feature sensitivity ranking
        if self.is_classifier:
            # Use probability change as sensitivity measure if available
            if "probability_change_mean" in results_df.columns:
                sensitivity_col = "probability_change_mean"
            else:
                sensitivity_col = "prediction_changes"
        else:
            sensitivity_col = "prediction_change_mean"

        # Group by feature and calculate mean sensitivity
        feature_sensitivity = (
            results_df.groupby("feature")[sensitivity_col].mean().reset_index()
        )
        feature_sensitivity.columns = ["feature", "sensitivity"]
        feature_sensitivity = feature_sensitivity.sort_values(
            "sensitivity", ascending=False
        )

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Performance change vs perturbation scale for each feature
        for feature in perturb_features:
            feature_data = results_df[results_df["feature"] == feature]

            if self.is_classifier:
                if "accuracy_change" in feature_data.columns:
                    y_metric = "accuracy_change"
                    y_label = "Accuracy Decrease"
                else:
                    y_metric = "prediction_changes"
                    y_label = "Prediction Changes"
            else:
                y_metric = "mse_change"
                y_label = "MSE Increase"

            axes[0].plot(
                feature_data["perturbation_scale"],
                feature_data[y_metric],
                "o-",
                label=feature,
            )

        axes[0].set_xlabel("Perturbation Scale")
        axes[0].set_ylabel(y_label)
        axes[0].set_title("Model Robustness to Feature Perturbations")
        if len(perturb_features) <= 10:  # Only show legend if not too many features
            axes[0].legend()

        # Plot 2: Feature sensitivity ranking
        y_pos = np.arange(len(feature_sensitivity))
        axes[1].barh(y_pos, feature_sensitivity["sensitivity"])
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(feature_sensitivity["feature"])
        axes[1].invert_yaxis()  # Labels read top-to-bottom
        axes[1].set_xlabel("Sensitivity")
        axes[1].set_title("Feature Sensitivity Ranking")

        plt.tight_layout()

        return {
            "plot": fig,
            "perturbations": results_df,
            "feature_sensitivity": feature_sensitivity,
        }

    def diagnose_resilience(
        self,
        challenge_fraction: float = 0.2,
        n_challenges: int = 5,
        dataset: str = "test",
        random_state: int = None,
    ):
        """
        Evaluate model resilience by analyzing performance on challenging data subsets.

        This function identifies challenging data subsets where the model performs poorly
        and compares performance on these subsets to overall performance. It helps understand
        the model's resilience to difficult cases.

        Parameters
        ----------
        challenge_fraction : float, default=0.2
            Fraction of data to include in each challenging subset

        n_challenges : int, default=5
            Number of different challenging subsets to generate

        dataset : {"main", "train", "test"}, default="test"
            Which dataset partition to use for the analysis

        random_state : int, default=None
            Random seed for reproducible results. If None, uses the TestSuite's random_state.

        Returns
        -------
        dict
            A dictionary containing the resilience analysis results with the following keys:
            - plot: A matplotlib figure showing performance on challenging subsets
            - challenges: A pandas DataFrame with performance on each challenging subset
            - overall: Overall performance metrics on the dataset
        """
        self._validate_setup()

        # Set random state if not provided
        if random_state is None:
            random_state = self.random_state

        # Get the specified dataset
        df = self._get_dataset(dataset)

        X = df[self.feature_cols]
        y_true = df[self.target_col]

        # Make predictions on the entire dataset
        if self.is_classifier:
            y_pred = self.model.predict(X)

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            # Compute overall metrics
            overall_metrics = {
                "dataset": "overall",
                "subset": "all",
                "size": len(df),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

            # Compute metrics for random subsets as baseline
            random_metrics = []
            random_subset_size = int(challenge_fraction * len(df))

            for i in range(n_challenges):
                # Generate random subset
                np.random.seed(random_state + i)
                random_idx = np.random.choice(
                    len(df), size=random_subset_size, replace=False
                )

                # Compute metrics
                random_metrics.append(
                    {
                        "dataset": "random",
                        "subset": f"random_{i+1}",
                        "size": random_subset_size,
                        "accuracy": accuracy_score(
                            y_true.iloc[random_idx], y_pred[random_idx]
                        ),
                        "precision": precision_score(
                            y_true.iloc[random_idx], y_pred[random_idx], zero_division=0
                        ),
                        "recall": recall_score(
                            y_true.iloc[random_idx], y_pred[random_idx], zero_division=0
                        ),
                        "f1": f1_score(
                            y_true.iloc[random_idx], y_pred[random_idx], zero_division=0
                        ),
                    }
                )

            # Compute instance-level error (for identifying challenging cases)
            error = y_pred != y_true

            # Generate challenging subsets based on prediction errors
            challenge_metrics = []

            # Challenge 1: Instances with highest prediction error
            challenge_idx = np.where(error)[0]
            if len(challenge_idx) > random_subset_size:
                challenge_idx = challenge_idx[:random_subset_size]

            if len(challenge_idx) > 0:
                challenge_metrics.append(
                    {
                        "dataset": "challenge",
                        "subset": "highest_error",
                        "size": len(challenge_idx),
                        "accuracy": accuracy_score(
                            y_true.iloc[challenge_idx], y_pred[challenge_idx]
                        ),
                        "precision": precision_score(
                            y_true.iloc[challenge_idx],
                            y_pred[challenge_idx],
                            zero_division=0,
                        ),
                        "recall": recall_score(
                            y_true.iloc[challenge_idx],
                            y_pred[challenge_idx],
                            zero_division=0,
                        ),
                        "f1": f1_score(
                            y_true.iloc[challenge_idx],
                            y_pred[challenge_idx],
                            zero_division=0,
                        ),
                    }
                )

            # Challenge 2: Class imbalance - overrepresent minority class
            class_counts = np.bincount(y_true)
            minority_class = np.argmin(class_counts)
            minority_idx = np.where(y_true == minority_class)[0]

            # Create a subset with high proportion of minority class
            if len(minority_idx) > 0:
                # Use all minority class instances
                imbalance_idx = list(minority_idx)

                # Add some majority class instances if needed
                if len(imbalance_idx) < random_subset_size:
                    majority_idx = np.where(y_true != minority_class)[0]
                    n_to_add = min(
                        random_subset_size - len(imbalance_idx), len(majority_idx)
                    )
                    if n_to_add > 0:
                        imbalance_idx.extend(
                            np.random.choice(majority_idx, size=n_to_add, replace=False)
                        )

                challenge_metrics.append(
                    {
                        "dataset": "challenge",
                        "subset": "class_imbalance",
                        "size": len(imbalance_idx),
                        "accuracy": accuracy_score(
                            y_true.iloc[imbalance_idx], y_pred[imbalance_idx]
                        ),
                        "precision": precision_score(
                            y_true.iloc[imbalance_idx],
                            y_pred[imbalance_idx],
                            zero_division=0,
                        ),
                        "recall": recall_score(
                            y_true.iloc[imbalance_idx],
                            y_pred[imbalance_idx],
                            zero_division=0,
                        ),
                        "f1": f1_score(
                            y_true.iloc[imbalance_idx],
                            y_pred[imbalance_idx],
                            zero_division=0,
                        ),
                    }
                )

            # Challenge 3: Feature outliers
            outlier_scores = np.zeros(len(df))

            # Compute outlier score based on feature z-scores
            for feature in self.feature_cols:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    z_scores = np.abs(
                        (df[feature] - df[feature].mean()) / df[feature].std()
                    )
                    outlier_scores += z_scores

            # Select instances with highest outlier scores
            outlier_idx = np.argsort(outlier_scores)[-random_subset_size:]

            challenge_metrics.append(
                {
                    "dataset": "challenge",
                    "subset": "feature_outliers",
                    "size": len(outlier_idx),
                    "accuracy": accuracy_score(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx]
                    ),
                    "precision": precision_score(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx], zero_division=0
                    ),
                    "recall": recall_score(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx], zero_division=0
                    ),
                    "f1": f1_score(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx], zero_division=0
                    ),
                }
            )

            # Combine all metrics
            all_metrics = [overall_metrics] + random_metrics + challenge_metrics
            metrics_df = pd.DataFrame(all_metrics)

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))

            # Group by dataset type and subset
            dataset_types = metrics_df["dataset"].unique()
            metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

            # Plot bars for each metric and dataset type
            bar_width = 0.2
            x = np.arange(len(metrics_to_plot))

            for i, dataset_type in enumerate(dataset_types):
                dataset_metrics = metrics_df[metrics_df["dataset"] == dataset_type]

                if dataset_type == "overall":
                    # Just one bar for overall
                    values = [
                        dataset_metrics[metric].values[0] for metric in metrics_to_plot
                    ]
                    ax.bar(x + i * bar_width, values, width=bar_width, label=f"Overall")
                else:
                    # Average across subsets for this dataset type
                    values = [
                        dataset_metrics[metric].mean() for metric in metrics_to_plot
                    ]
                    ax.bar(
                        x + i * bar_width,
                        values,
                        width=bar_width,
                        label=f"{dataset_type.title()}",
                    )

                    # Add error bars if multiple subsets
                    if len(dataset_metrics) > 1:
                        errors = [
                            dataset_metrics[metric].std() for metric in metrics_to_plot
                        ]
                        ax.errorbar(
                            x + i * bar_width,
                            values,
                            yerr=errors,
                            fmt="none",
                            ecolor="black",
                        )

            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylabel("Score")
            ax.set_title("Model Performance Across Different Data Subsets")
            ax.legend()

            plt.tight_layout()

        else:
            # For regression models
            y_pred = self.model.predict(X)

            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            # Compute overall metrics
            overall_metrics = {
                "dataset": "overall",
                "subset": "all",
                "size": len(df),
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
            }

            # Compute metrics for random subsets as baseline
            random_metrics = []
            random_subset_size = int(challenge_fraction * len(df))

            for i in range(n_challenges):
                # Generate random subset
                np.random.seed(random_state + i)
                random_idx = np.random.choice(
                    len(df), size=random_subset_size, replace=False
                )

                # Compute metrics
                random_metrics.append(
                    {
                        "dataset": "random",
                        "subset": f"random_{i+1}",
                        "size": random_subset_size,
                        "mse": mean_squared_error(
                            y_true.iloc[random_idx], y_pred[random_idx]
                        ),
                        "mae": mean_absolute_error(
                            y_true.iloc[random_idx], y_pred[random_idx]
                        ),
                        "r2": r2_score(y_true.iloc[random_idx], y_pred[random_idx]),
                    }
                )

            # Compute instance-level error
            error = np.abs(y_pred - y_true)

            # Generate challenging subsets
            challenge_metrics = []

            # Challenge 1: Instances with highest prediction error
            challenge_idx = np.argsort(error)[-random_subset_size:]

            challenge_metrics.append(
                {
                    "dataset": "challenge",
                    "subset": "highest_error",
                    "size": len(challenge_idx),
                    "mse": mean_squared_error(
                        y_true.iloc[challenge_idx], y_pred[challenge_idx]
                    ),
                    "mae": mean_absolute_error(
                        y_true.iloc[challenge_idx], y_pred[challenge_idx]
                    ),
                    "r2": r2_score(y_true.iloc[challenge_idx], y_pred[challenge_idx]),
                }
            )

            # Challenge 2: Target outliers
            y_z_scores = np.abs((y_true - y_true.mean()) / y_true.std())
            outlier_idx = np.argsort(y_z_scores)[-random_subset_size:]

            challenge_metrics.append(
                {
                    "dataset": "challenge",
                    "subset": "target_outliers",
                    "size": len(outlier_idx),
                    "mse": mean_squared_error(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx]
                    ),
                    "mae": mean_absolute_error(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx]
                    ),
                    "r2": r2_score(y_true.iloc[outlier_idx], y_pred[outlier_idx]),
                }
            )

            # Challenge 3: Feature outliers
            outlier_scores = np.zeros(len(df))

            # Compute outlier score based on feature z-scores
            for feature in self.feature_cols:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    z_scores = np.abs(
                        (df[feature] - df[feature].mean()) / df[feature].std()
                    )
                    outlier_scores += z_scores

            # Select instances with highest outlier scores
            outlier_idx = np.argsort(outlier_scores)[-random_subset_size:]

            challenge_metrics.append(
                {
                    "dataset": "challenge",
                    "subset": "feature_outliers",
                    "size": len(outlier_idx),
                    "mse": mean_squared_error(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx]
                    ),
                    "mae": mean_absolute_error(
                        y_true.iloc[outlier_idx], y_pred[outlier_idx]
                    ),
                    "r2": r2_score(y_true.iloc[outlier_idx], y_pred[outlier_idx]),
                }
            )

            # Combine all metrics
            all_metrics = [overall_metrics] + random_metrics + challenge_metrics
            metrics_df = pd.DataFrame(all_metrics)

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))

            # Group by dataset type and subset
            dataset_types = metrics_df["dataset"].unique()
            metrics_to_plot = ["mse", "mae"]  # Excluding r2 which can be negative

            # Plot bars for each metric and dataset type
            bar_width = 0.2
            x = np.arange(len(metrics_to_plot))

            for i, dataset_type in enumerate(dataset_types):
                dataset_metrics = metrics_df[metrics_df["dataset"] == dataset_type]

                if dataset_type == "overall":
                    # Just one bar for overall
                    values = [
                        dataset_metrics[metric].values[0] for metric in metrics_to_plot
                    ]
                    ax.bar(x + i * bar_width, values, width=bar_width, label=f"Overall")
                else:
                    # Average across subsets for this dataset type
                    values = [
                        dataset_metrics[metric].mean() for metric in metrics_to_plot
                    ]
                    ax.bar(
                        x + i * bar_width,
                        values,
                        width=bar_width,
                        label=f"{dataset_type.title()}",
                    )

                    # Add error bars if multiple subsets
                    if len(dataset_metrics) > 1:
                        errors = [
                            dataset_metrics[metric].std() for metric in metrics_to_plot
                        ]
                        ax.errorbar(
                            x + i * bar_width,
                            values,
                            yerr=errors,
                            fmt="none",
                            ecolor="black",
                        )

            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylabel("Error")
            ax.set_title("Model Error Across Different Data Subsets")
            ax.legend()

            plt.tight_layout()

        return {"plot": fig, "challenges": metrics_df, "overall": overall_metrics}
