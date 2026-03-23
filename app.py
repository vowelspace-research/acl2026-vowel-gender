"""
Gender Classification from Speaker-Normalized Vowel Formants
ACL Rolling Review Submission Code
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#data loading

def load_data(path="speaker_data.csv"):
    df = pd.read_csv(path)

    # Speaker normalization (z-score within speaker)
    df["F1_norm"] = df.groupby("Speaker")["F1"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df["F2_norm"] = df.groupby("Speaker")["F2"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    df = df.dropna().reset_index(drop=True)
    df["Gender_binary"] = df["Gender"].map({"Female": 0, "Male": 1})

    return df


#mixed effects modeling 

def run_mixed_effects(df):
    model_f1 = smf.mixedlm("F1_norm ~ Gender", df, groups=df["Speaker"])
    model_f2 = smf.mixedlm("F2_norm ~ Gender", df, groups=df["Speaker"])

    return model_f1.fit(), model_f2.fit()


#leave-one-speaker-out classification

def loso_classification(df):
    X = df[["F1_norm", "F2_norm", "F3"]]
    y = df["Gender_binary"]
    groups = df["Speaker"]

    logo = LeaveOneGroupOut()
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )

    all_preds = []
    all_probs = []
    all_true = []

    for train_idx, test_idx in logo.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_true.extend(y.iloc[test_idx])

    accuracy = np.mean(np.array(all_preds) == np.array(all_true))
    auc_score = roc_auc_score(all_true, all_probs)
    cm = confusion_matrix(all_true, all_preds)

    return accuracy, auc_score, cm


#permutation test

def permutation_test(df, true_accuracy, n_permutations=100):
    X = df[["F1_norm", "F2_norm", "F3"]]
    y = df["Gender_binary"].values
    groups = df["Speaker"]

    logo = LeaveOneGroupOut()
    perm_scores = []

    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        fold_scores = []

        for train_idx, test_idx in logo.split(X, y_perm, groups):
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            )
            model.fit(X.iloc[train_idx], y_perm[train_idx])
            score = model.score(X.iloc[test_idx], y_perm[test_idx])
            fold_scores.append(score)

        perm_scores.append(np.mean(fold_scores))

    p_value = np.mean(np.array(perm_scores) >= true_accuracy)
    return p_value


#within-vowel LOSO

def within_vowel_loso(df):
    results = {}

    for vowel in df["Vowel"].unique():
        subset = df[df["Vowel"] == vowel]

        X = subset[["F1_norm", "F2_norm", "F3"]]
        y = subset["Gender_binary"]
        groups = subset["Speaker"]

        logo = LeaveOneGroupOut()
        scores = []

        for train_idx, test_idx in logo.split(X, y, groups):
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))

        results[vowel] = np.mean(scores)

    return results


#main

if __name__ == "__main__":

    df = load_data()

    # Mixed effects
    result_f1, result_f2 = run_mixed_effects(df)

    # LOSO classification
    accuracy, auc_score, cm = loso_classification(df)

    # Permutation test
    p_value = permutation_test(df, accuracy)

    # Within-vowel analysis
    vowel_results = within_vowel_loso(df)

    print("\nLOSO Accuracy:", round(accuracy, 3))
    print("AUC:", round(auc_score, 3))
    print("Permutation p-value:", round(p_value, 4))
    print("Confusion Matrix:\n", cm)

    print("\nPer-Vowel LOSO Accuracy:")
    for vowel, score in vowel_results.items():
        print(f"{vowel}: {round(score, 3)}")
