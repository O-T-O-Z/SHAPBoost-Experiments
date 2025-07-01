import numpy as np
import pandas as pd
from pycox.datasets import metabric
from shap.datasets import nhanesi
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import OneHotEncoder
from sksurv.datasets import load_aids, load_breast_cancer, load_whas500
from ucimlrepo import fetch_ucirepo

pd.set_option("future.no_silent_downcasting", True)


def prepare_nacd() -> None:
    """Prepare the NACD dataset from the PSSP website."""
    df = pd.read_csv(
        "http://pssp.srv.ualberta.ca/system/predictors/datasets/000/000/032/original/All_Data_updated_may2011_CLEANED.csv?1350302245"  # noqa
    )
    X_nacd = df.drop(["SURVIVAL", "CENSORED"], axis=1)
    y_nacd = df[["SURVIVAL", "CENSORED"]].copy()
    y_nacd.loc[:, "lower_bound"] = y_nacd["SURVIVAL"]
    y_nacd.loc[:, "upper_bound"] = y_nacd["SURVIVAL"]
    y_nacd.loc[y_nacd["CENSORED"] == 1, "upper_bound"] = np.inf
    y_nacd = y_nacd.drop(["SURVIVAL", "CENSORED"], axis=1)
    df = pd.concat([X_nacd, y_nacd], axis=1)
    df.to_csv("datasets/nacd_cleaned.csv", index=False)


def prepare_support() -> None:
    """Prepare the SUPPORT dataset."""
    FILL_VALUES = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9.0,
        "urine": 2502.0,
    }

    TO_DROP = [
        "aps",
        "sps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "sfdm2",
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
    ]

    # load, drop columns, fill using specified fill values
    df = (
        pd.read_csv("raw_datasets/support2.csv")
        .drop(TO_DROP, axis=1)
        .fillna(value=FILL_VALUES)
    )
    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    object_columns = df.drop(df.select_dtypes(exclude="object").columns, axis=1)
    df.drop(object_columns.columns, axis=1, inplace=True)
    df = pd.concat([df, enc.fit_transform(object_columns)], axis=1)

    df = df.fillna(df.median())
    X_support = df.drop(["death", "d.time"], axis=1)
    X_support = X_support.replace(True, 1).replace(False, 0)

    y_support = df[["death", "d.time"]]
    y_support = y_support.copy()
    y_support.loc[:, "lower_bound"] = y_support["d.time"].astype(float)
    y_support.loc[:, "upper_bound"] = y_support["d.time"].astype(float)
    y_support.loc[y_support["death"] == 1, "upper_bound"] = np.inf
    y_support = y_support.drop(["death", "d.time"], axis=1)

    # combine the two datasets
    df = pd.concat([X_support, y_support], axis=1)
    df.to_csv("datasets/support_cleaned.csv", index=False)


def prepare_nhanes() -> None:
    """Prepare the NHANES dataset."""
    X, y = nhanesi()
    y_lower_bound = abs(y)
    y_upper_bound = np.array([np.inf if i < 0 else i for i in y])
    y = pd.DataFrame(
        np.array([y_lower_bound, y_upper_bound]).T,
        columns=["lower_bound", "upper_bound"],
        index=X.index,
    )
    # fill missing values with median
    X = X.replace(True, 1).replace(False, 0)
    X = X.fillna(X.median())

    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/nhanes_cleaned.csv", index=False)


def prepare_metabric() -> None:
    """Prepare the METABRIC dataset."""
    df = metabric.read_df()
    X = df.drop(columns=["duration", "event"])
    y = df[["duration", "event"]]
    y = y.copy()
    y["lower_bound"] = y["duration"]
    y["upper_bound"] = y["duration"]
    y.loc[y["event"] == 0, "upper_bound"] = np.inf
    y = y.drop(columns=["duration", "event"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/metabric_cleaned.csv", index=False)


def prepare_metabric_full() -> None:
    """Prepare the METABRIC dataset with all gene expression data."""
    df = pd.read_csv(
        "raw_datasets/METABRIC_RNA_Mutation.csv"
    )  # from https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
    drop_columns = [
        "patient_id",
        # "age_at_diagnosis",
        "type_of_breast_surgery",
        "cancer_type",
        "cancer_type_detailed",
        "cellularity",
        # "chemotherapy",
        "cohort",
        "pam50_+_claudin-low_subtype",
        "er_status_measured_by_ihc",
        # "er_status",
        "neoplasm_histologic_grade",
        "her2_status_measured_by_snp6",
        "her2_status",
        "tumor_other_histologic_subtype",
        # "hormone_therapy",
        "inferred_menopausal_state",
        "integrative_cluster",
        "primary_tumor_laterality",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "nottingham_prognostic_index",
        "oncotree_code",
        "pr_status",
        # "radio_therapy",
        "tumor_size",
        "tumor_stage",
        "death_from_cancer",
        "3-gene_classifier_subtype",
        "siah1_mut",  # highly correlated with outcome
    ]
    df = df.drop(columns=drop_columns)
    df["er_status"] = df["er_status"].map({"Positive": 1, "Negative": 0}).astype(str)
    for col in df.select_dtypes(include="object").columns:
        if col.endswith("mut"):
            # transform such that everything is 0 except if it is a string
            df[col] = df[col].apply(lambda x: 1 if x != "0" else 0)
            df[col] = df[col].astype(int)

    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    object_columns = df.drop(df.select_dtypes(exclude="object").columns, axis=1).astype(
        str
    )
    df.drop(object_columns.columns, axis=1, inplace=True)
    df = pd.concat([df, enc.fit_transform(object_columns)], axis=1)

    y = df[["overall_survival_months", "overall_survival"]]
    X = df.drop(columns=["overall_survival_months", "overall_survival"])
    y = y.copy()
    y["lower_bound"] = y["overall_survival_months"]
    y["upper_bound"] = y["overall_survival_months"]
    y.loc[y["overall_survival"] == 1, "upper_bound"] = np.inf
    y = y.drop(columns=["overall_survival_months", "overall_survival"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/metabric_full_cleaned.csv", index=False)


def prepare_metabric_regression() -> None:
    """Prepare the METABRIC dataset for regression."""
    df = pd.read_csv(
        "raw_datasets/METABRIC_RNA_Mutation.csv"
    )  # from https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
    drop_columns = [
        "patient_id",
        # "age_at_diagnosis",
        "type_of_breast_surgery",
        "cancer_type",
        "cancer_type_detailed",
        "cellularity",
        # "chemotherapy",
        "cohort",
        "pam50_+_claudin-low_subtype",
        "er_status_measured_by_ihc",
        # "er_status",
        "neoplasm_histologic_grade",
        "her2_status_measured_by_snp6",
        "her2_status",
        "tumor_other_histologic_subtype",
        # "hormone_therapy",
        "inferred_menopausal_state",
        "integrative_cluster",
        "primary_tumor_laterality",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "oncotree_code",
        "pr_status",
        # "radio_therapy",
        "tumor_size",
        "tumor_stage",
        "death_from_cancer",
        "3-gene_classifier_subtype",
        "overall_survival_months",
        "overall_survival",
    ]
    df = df.drop(columns=drop_columns)
    df["er_status"] = df["er_status"].map({"Positive": 1, "Negative": 0}).astype(str)
    for col in df.select_dtypes(include="object").columns:
        if col.endswith("mut"):
            df[col] = df[col].apply(lambda x: 1 if x != "0" else 0)
            df[col] = df[col].astype(int)

    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    object_columns = df.drop(df.select_dtypes(exclude="object").columns, axis=1).astype(
        str
    )
    df.drop(object_columns.columns, axis=1, inplace=True)
    df = pd.concat([df, enc.fit_transform(object_columns)], axis=1)
    y = df["nottingham_prognostic_index"]
    X = df.drop(columns=["nottingham_prognostic_index"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/metabric_regression_cleaned.csv", index=False)


def prepare_eyedata() -> None:
    """Prepare the eye data dataset."""
    df = pd.read_csv("raw_datasets/eyedata.csv")
    X = df.drop(columns=["trim32", "Unnamed: 0"])
    y = df["trim32"]
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/eyedata_cleaned.csv", index=False)


def prepare_diabetes() -> None:
    """Prepare the diabetes dataset."""
    data = load_diabetes()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])  # type: ignore
    y = pd.DataFrame(data["target"], columns=["target"])  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/diabetes_cleaned.csv", index=False)


def prepare_housing() -> None:
    """Prepare the housing dataset."""
    data = fetch_california_housing()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])  # type: ignore
    y = pd.DataFrame(data["target"], columns=data["target_names"])  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/housing_cleaned.csv", index=False)


def prepare_crime() -> None:
    """Prepare the crime dataset."""
    communities_and_crime_unnormalized = fetch_ucirepo(id=211)
    X = communities_and_crime_unnormalized.data.features  # type: ignore
    y = communities_and_crime_unnormalized.data.targets  # type: ignore

    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    object_columns = X.drop(X.select_dtypes(exclude="object").columns, axis=1)
    X.drop(object_columns.columns, axis=1, inplace=True)
    X = pd.concat([X, enc.fit_transform(object_columns)], axis=1)

    X = X.astype(float)
    X = X.fillna(X.median())
    y = y["violentPerPop"]
    y = y.astype(float)
    y = y.dropna()
    X = X.loc[y.index]
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/crime_cleaned.csv", index=False)


def prepare_parkinsons() -> None:
    """Prepare the parkinsons dataset."""
    parkinsons_telemonitoring = fetch_ucirepo(id=189)
    X = parkinsons_telemonitoring.data.features  # type: ignore
    y = parkinsons_telemonitoring.data.targets["total_UPDRS"]  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/parkinsons_cleaned.csv", index=False)


def prepare_msd() -> None:
    """Prepare the Million Song Dataset."""
    data = pd.read_csv("raw_datasets/YearPredictionMSD.txt", header=None)
    y = data.pop(0)  # type: ignore
    X = data
    df = pd.concat([X, y], axis=1)
    df.sample(10000, random_state=42).to_csv("datasets/msd_cleaned.csv", index=False)


def prepare_whas500() -> None:
    """Prepare the WHAS500 dataset."""
    X, y_raw = load_whas500()
    for col in X.select_dtypes(include="category").columns:
        X[col] = X[col].astype(int)
    y = pd.DataFrame(y_raw)
    y["lower_bound"] = y["lenfol"]
    y["upper_bound"] = y["lenfol"]
    y.loc[y["fstat"] == 0, "upper_bound"] = np.inf
    y = y.drop(columns=["fstat", "lenfol"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/whas500_cleaned.csv", index=False)


def prepare_breast_cancer() -> None:
    """Prepare the breast cancer dataset."""
    X, y_raw = load_breast_cancer()

    category_columns = X.drop(X.select_dtypes(exclude="category").columns, axis=1)
    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    X.drop(category_columns.columns, axis=1, inplace=True)
    X = pd.concat([X, enc.fit_transform(category_columns)], axis=1)
    y = pd.DataFrame(y_raw)
    y["lower_bound"] = y["t.tdm"]
    y["upper_bound"] = y["t.tdm"]
    y.loc[y["e.tdm"] == 0, "upper_bound"] = np.inf
    y = y.drop(columns=["t.tdm", "e.tdm"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/breast_cancer_cleaned.csv", index=False)


def prepare_aids() -> None:
    """Prepare the AIDS dataset."""
    X, y_raw = load_aids()
    y = pd.DataFrame(y_raw)
    category_columns = X.drop(X.select_dtypes(exclude="category").columns, axis=1)
    enc = OneHotEncoder(drop="if_binary", sparse_output=False).set_output(
        transform="pandas"
    )
    X.drop(category_columns.columns, axis=1, inplace=True)
    X = pd.concat([X, enc.fit_transform(category_columns)], axis=1)
    X.drop(["txgrp_3", "txgrp_4"], axis=1, inplace=True)
    y["lower_bound"] = y["time"]
    y["upper_bound"] = y["time"]
    y.loc[y["censor"] == 1, "upper_bound"] = np.inf
    y = y.drop(columns=["time", "censor"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/aids_cleaned.csv", index=False)


def load_regression_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a regression dataset from the datasets folder.

    :param name: The name of the dataset to load.
    :return: A tuple containing the features and the target.
    """
    df = pd.read_csv(f"datasets/{name}_cleaned.csv")
    if name == "crime":
        target = "violentPerPop"
    elif name == "diabetes":
        target = "target"
    elif name == "housing":
        target = "MedHouseVal"
    elif name == "msd":
        target = "0"
    elif name == "parkinsons":
        target = "total_UPDRS"
    elif name == "metabric_regression":
        target = "nottingham_prognostic_index"
    elif name == "eyedata":
        target = "trim32"

    X = df.drop(target, axis=1)
    y = df[[target]]
    return X, y


def load_survival_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a survival dataset from the datasets folder.

    :param name: The name of the dataset to load.
    :return: A tuple containing the features and the target.
    """
    df = pd.read_csv(f"datasets/{name}_cleaned.csv")
    X = df.drop(["lower_bound", "upper_bound"], axis=1)
    y = df[["lower_bound", "upper_bound"]]
    return X, y
