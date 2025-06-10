import numpy as np
from collections import Counter
from sklearn.utils import _check_X_y

# Import imblearn over-sampling algorithms
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# Import imblearn under-sampling algorithms
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks

class ImbalancedSampler:
    """
    A class to perform upsampling or downsampling on imbalanced datasets
    using various algorithms from the imblearn library.

    The sampling strategy is controlled by 'target_ratio', which defines the
    ratio of the minority class to the majority class *after* resampling.

    Attributes:
        algorithm (str): The name of the imblearn algorithm to use (e.g., 'SMOTE',
                         'RandomUnderSampler', 'NearMiss-1').
        sampler_type (str): Specifies the type of sampling: 'upsample' or 'downsample'.
        target_ratio (float): The desired ratio of the minority class to the majority
                              class after resampling. For example, a target_ratio of
                              1.0 means the minority class will have the same number
                              of samples as the majority class. A target_ratio of 0.5
                              means the minority class will have 50% of the majority
                              class's sample count.
        random_state (int): Seed for the random number generator for reproducibility.
        _sampler (object): The instantiated imblearn sampler object.
    """

    def __init__(self, algorithm: str, sampler_type: str, target_ratio: float = 1.0, random_state: int = 42):
        """
        Initializes the ImbalancedSampler with the specified algorithm and parameters.

        Args:
            algorithm (str): The name of the imblearn algorithm to use.
                             Accepted upsampling algorithms: 'RandomOverSampler', 'SMOTE', 'ADASYN'.
                             Accepted downsampling algorithms: 'RandomUnderSampler',
                                                              'NearMiss-1', 'NearMiss-2', 'NearMiss-3',
                                                              'CondensedNearestNeighbour',
                                                              'EditedNearestNeighbours', 'TomekLinks'.
            sampler_type (str): Specifies the type of sampling: 'upsample' or 'downsample'.
            target_ratio (float): The desired ratio of the minority class to the majority
                                  class after resampling. Default is 1.0 (balanced).
            random_state (int): Seed for the random number generator for reproducibility.
        """
        self.algorithm = algorithm
        self.sampler_type = sampler_type.lower()
        self.target_ratio = target_ratio
        self.random_state = random_state
        self._sampler = None

        self._initialize_sampler()

    def _initialize_sampler(self):
        """
        Internal method to initialize the appropriate imblearn sampler based on
        the chosen algorithm and type.
        """
        # The sampling_strategy parameter for imblearn often accepts a float
        # indicating the ratio of the number of samples of the minority class
        # over the number of samples of the majority class after resampling.
        sampling_strategy_param = self.target_ratio

        if self.sampler_type == 'upsample':
            if self.algorithm == 'RandomOverSampler':
                self._sampler = RandomOverSampler(sampling_strategy=sampling_strategy_param,
                                                  random_state=self.random_state)
            elif self.algorithm == 'SMOTE':
                self._sampler = SMOTE(sampling_strategy=sampling_strategy_param,
                                      random_state=self.random_state)
            elif self.algorithm == 'ADASYN':
                self._sampler = ADASYN(sampling_strategy=sampling_strategy_param,
                                       random_state=self.random_state)
            else:
                raise ValueError(
                    f"Unsupported upsampling algorithm: {self.algorithm}. "
                    "Choose from 'RandomOverSampler', 'SMOTE', 'ADASYN'."
                )
        elif self.sampler_type == 'downsample':
            if self.algorithm == 'RandomUnderSampler':
                self._sampler = RandomUnderSampler(sampling_strategy=sampling_strategy_param,
                                                   random_state=self.random_state)
            elif self.algorithm.startswith('NearMiss'):
                version = int(self.algorithm.split('-')[1]) if '-' in self.algorithm else 1
                if version not in [1, 2, 3]:
                    raise ValueError(f"Invalid NearMiss version: {version}. Choose 1, 2, or 3.")
                self._sampler = NearMiss(version=version, sampling_strategy=sampling_strategy_param,
                                         random_state=self.random_state)
            elif self.algorithm == 'CondensedNearestNeighbour':
                # CNN often takes a long time and might not work well with direct ratio for small datasets
                # Defaulting to 'majority' or 'not minority' for sampling strategy if ratio is not 1.0,
                # as float strategy often means 'the ratio of minority class to majority class after resampling'.
                # For CNN, it means keeping majority class samples which are close to minority class.
                # Here, we'll still use the float if provided.
                self._sampler = CondensedNearestNeighbour(sampling_strategy=sampling_strategy_param,
                                                         random_state=self.random_state)
            elif self.algorithm == 'EditedNearestNeighbours':
                self._sampler = EditedNearestNeighbours(sampling_strategy=sampling_strategy_param)
            elif self.algorithm == 'TomekLinks':
                # TomekLinks only removes links, it doesn't change the overall ratio in the same way
                # as other methods. The sampling_strategy often refers to which classes to resample.
                # We'll apply it to 'not minority' by default if target_ratio is 1.0
                # For other ratios, imblearn's handling of sampling_strategy for TomekLinks
                # might be different from simple over/under samplers.
                # If target_ratio is used, it will be applied to the 'not minority' classes
                # so that they have the specified ratio *relative to the minority class*.
                self._sampler = TomekLinks(sampling_strategy=sampling_strategy_param)
            else:
                raise ValueError(
                    f"Unsupported downsampling algorithm: {self.algorithm}. "
                    "Choose from 'RandomUnderSampler', 'NearMiss-1', 'NearMiss-2', 'NearMiss-3', "
                    "'CondensedNearestNeighbour', 'EditedNearestNeighbours', 'TomekLinks'."
                )
        else:
            raise ValueError(
                f"Unsupported sampler_type: {self.sampler_type}. Choose 'upsample' or 'downsample'."
            )

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        """
        Resamples the input dataset (X, y) based on the initialized sampler.

        Args:
            X (np.ndarray): The features of the dataset.
            y (np.ndarray): The target variable of the dataset.

        Returns:
            tuple: A tuple (X_resampled, y_resampled) containing the resampled
                   features and target variable.
        """
        if self._sampler is None:
            raise RuntimeError("Sampler not initialized. Call __init__ first.")

        # Ensure X and y are properly formatted (e.g., numpy arrays)
        X, y = _check_X_y(X, y)

        print(f"Original dataset shape: {Counter(y)}")
        try:
            X_resampled, y_resampled = self._sampler.fit_resample(X, y)
            print(f"Resampled dataset shape using {self.algorithm}: {Counter(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"An error occurred during resampling with {self.algorithm}: {e}")
            print("Please check if the algorithm is suitable for your dataset characteristics.")
            # Re-raise the exception or handle it gracefully based on requirements
            raise


# --- Example Usage ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from collections import Counter

    # Generate a synthetic imbalanced dataset
    # We'll create a binary classification problem with 90% in class 0 and 10% in class 1
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.9, 0.1],
                               flip_y=0, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n--- Original Training Dataset Distribution ---")
    print(f"Original training dataset shape: {Counter(y_train)}")

    print("\n--- Upsampling Examples ---")

    # Example 1: Upsample minority class using SMOTE to have 50% of the majority class count
    # Original minority class is 80. Majority is 720. Target ratio 0.5 means minority will become 0.5 * 720 = 360.
    print("\nAttempting SMOTE upsampling to target ratio 0.5...")
    smote_sampler = ImbalancedSampler(algorithm='SMOTE', sampler_type='upsample', target_ratio=0.5, random_state=42)
    X_res_smote, y_res_smote = smote_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (SMOTE, target ratio 0.5): {Counter(y_res_smote)}")

    # Example 2: Upsample minority class using RandomOverSampler to achieve balanced classes (1:1 ratio)
    # Original minority class is 80. Majority is 720. Target ratio 1.0 means minority will become 1.0 * 720 = 720.
    print("\nAttempting RandomOverSampler upsampling to target ratio 1.0 (balanced)...")
    ros_sampler = ImbalancedSampler(algorithm='RandomOverSampler', sampler_type='upsample', target_ratio=1.0, random_state=42)
    X_res_ros, y_res_ros = ros_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (RandomOverSampler, target ratio 1.0): {Counter(y_res_ros)}")

    # Example 3: Upsample minority class using ADASYN to achieve 75% of the majority class count
    # Original minority class is 80. Majority is 720. Target ratio 0.75 means minority will become 0.75 * 720 = 540.
    print("\nAttempting ADASYN upsampling to target ratio 0.75...")
    adasyn_sampler = ImbalancedSampler(algorithm='ADASYN', sampler_type='upsample', target_ratio=0.75, random_state=42)
    X_res_adasyn, y_res_adasyn = adasyn_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (ADASYN, target ratio 0.75): {Counter(y_res_adasyn)}")


    print("\n--- Downsampling Examples ---")

    # Example 4: Downsample majority class using RandomUnderSampler to achieve balanced classes
    # Original minority class is 80. Majority is 720. Target ratio 1.0 means majority will be reduced to match minority, i.e., 80 samples.
    print("\nAttempting RandomUnderSampler downsampling to target ratio 1.0 (balanced)...")
    rus_sampler = ImbalancedSampler(algorithm='RandomUnderSampler', sampler_type='downsample', target_ratio=1.0, random_state=42)
    X_res_rus, y_res_rus = rus_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (RandomUnderSampler, target ratio 1.0): {Counter(y_res_rus)}")

    # Example 5: Downsample majority using NearMiss-1 to target ratio 0.8
    # Original minority class is 80. Majority is 720. Target ratio 0.8 means majority will be reduced so that minority is 0.8 * majority.
    # So, 80 = 0.8 * maj => maj = 80 / 0.8 = 100. Majority will be 100 samples.
    print("\nAttempting NearMiss-1 downsampling to target ratio 0.8...")
    nm1_sampler = ImbalancedSampler(algorithm='NearMiss-1', sampler_type='downsample', target_ratio=0.8, random_state=42)
    X_res_nm1, y_res_nm1 = nm1_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (NearMiss-1, target ratio 0.8): {Counter(y_res_nm1)}")

    # Example 6: Downsample using TomekLinks (often used for mild cleaning, target_ratio usually applies to 'not minority')
    print("\nAttempting TomekLinks downsampling (will remove overlapping instances)...")
    tl_sampler = ImbalancedSampler(algorithm='TomekLinks', sampler_type='downsample', target_ratio=1.0, random_state=42)
    X_res_tl, y_res_tl = tl_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (TomekLinks): {Counter(y_res_tl)}")

    # Example 7: Downsample using EditedNearestNeighbours
    print("\nAttempting EditedNearestNeighbours downsampling...")
    enn_sampler = ImbalancedSampler(algorithm='EditedNearestNeighbours', sampler_type='downsample', target_ratio=1.0, random_state=42)
    X_res_enn, y_res_enn = enn_sampler.fit_resample(X_train, y_train)
    print(f"Resampled dataset shape (EditedNearestNeighbours): {Counter(y_res_enn)}")

    # Example 8: Attempting an unsupported algorithm
    print("\n--- Testing Error Handling (Unsupported Algorithm) ---")
    try:
        invalid_sampler = ImbalancedSampler(algorithm='UnsupportedAlgorithm', sampler_type='upsample', target_ratio=1.0)
        invalid_sampler.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example 9: Attempting an unsupported sampler type
    print("\n--- Testing Error Handling (Unsupported Sampler Type) ---")
    try:
        invalid_type_sampler = ImbalancedSampler(algorithm='SMOTE', sampler_type='invalid', target_ratio=1.0)
        invalid_type_sampler.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"Caught expected error: {e}")