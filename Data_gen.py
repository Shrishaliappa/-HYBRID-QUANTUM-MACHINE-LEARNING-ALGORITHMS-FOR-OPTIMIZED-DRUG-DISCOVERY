from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE
from BCOA import brownian_cheetah_feature_selection
from Feature_extraction import extract_descriptors
from save_load import save
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def Data_gen():
    # === 1. Load Dataset ===
    file_path = r"C:\Users\ondezx\Downloads\3A4_test_disguised.csv\3A4_test_disguised.csv"
    df = pd.read_csv(file_path)
    df = df[:100]
    # === 2. Drop MOLECULE column ===
    df = df.drop(columns=["MOLECULE"])

    # === 3. Label 'Act' values into 3 classes ===
    def label_activity(act):
        if act <= 4.5:
            return 0  # Low
        elif act <= 5.5:
            return 1  # Medium
        else:
            return 2  # High

    df['Label'] = df['Act'].apply(label_activity)

    # === 4. Separate descriptors and labels ===
    X = df.drop(columns=["Act", "Label"])
    y = df["Label"]

    # === 5. Handle missing values ===
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # === 6. Normalize features ===
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # === 7. Outlier detection (DBSCAN) ===
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    outlier_labels = dbscan.fit_predict(X_scaled)

    # Filter non-outliers
    mask = outlier_labels != -1
    X_filtered = X_scaled[mask]
    y_filtered = y[mask].reset_index(drop=True)

    # === 8. Address Class Imbalance using SMOTE ===
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_filtered, y_filtered)

    # === 9. Save the final balanced dataset ===
    balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
    balanced_df['Label'] = y_balanced

    output_path = "qsar_dataset_balanced.csv"
    balanced_df.to_csv(output_path, index=False)

    # === 10. Display class distribution and sample ===
    print(" Class balancing complete. Final dataset saved to:", output_path)
    print("\nClass distribution after SMOTE:\n", y_balanced.value_counts())
    print("\nSample rows:\n", balanced_df.head())
    # Load the cleaned dataset (if not in memory)
    df = pd.read_csv("preprocessed_qsar_dataset.csv")  # or "qsar_dataset_balanced.csv"

    # Assume 'SMILES' column exists (you can add if needed)
    # Example: df['SMILES'] = ["CCO", "CCN", "CCC", ...]  # If not already present

    # Extract features
    descriptor_df = extract_descriptors(df['SMILES'])

    # Combine with label
    final_dataset = pd.concat([descriptor_df, df['Label']], axis=1)

    # Save final feature-rich dataset
    final_dataset.to_csv("qsar_final_descriptors.csv", index=False)

    print(" Feature extraction complete. Final dataset saved: qsar_final_descriptors.csv")
    print(final_dataset.head())
    # Load the dataset after descriptor extraction
    df = pd.read_csv("qsar_final_descriptors.csv")  # from previous step

    # Separate features and target
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # Call BCOA
    selected_features = brownian_cheetah_feature_selection(X, y)

    # Use only selected features
    X_selected = X[selected_features]
    final_df = pd.concat([X_selected, y], axis=1)

    # Save final dataset
    final_df.to_csv("qsar_selected_features.csv", index=False)

    print("Feature selection complete. Selected features saved: qsar_selected_features.csv")
    print(f"Selected features ({len(selected_features)}):", selected_features)


    # === 3.5 Load Feature-Selected Dataset ===
    df = pd.read_csv("qsar_selected_features.csv")

    # Separate features and labels
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # === Normalize the data so that sum of squares = 1 (required for amplitude encoding) ===
    X_normalized = normalize(X, norm='l2')  # L2 normalization makes sum(x_i^2) = 1

    # === Check normalization correctness (optional sanity check) ===
    assert np.allclose(np.sum(X_normalized ** 2, axis=1), 1.0, atol=1e-6), "Normalization failed"

    # === 3.6 Train/Test Splits ===

    # 70:30 split
    X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
        X_normalized, y, test_size=0.30, random_state=42, stratify=y
    )
    save("X_train_70", X_train_70)
    save("X_test_30", X_test_30)
    save("y_train_70", y_train_70)
    save("y_test_30", y_test_30)
    # 80:20 split
    X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
        X_normalized, y, test_size=0.20, random_state=42, stratify=y
    )

    save("X_train_80", X_train_80)
    save("X_test_20", X_test_20)
    save("y_train_80", y_train_80)
    save("y_test_20", y_test_20)