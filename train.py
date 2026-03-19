import os
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import config


def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    df = config.load_data()

    if config.WITH_GENDER:
        gender_counts = df["Sex"].value_counts()
        print("+-+" * 30)
        print("Gender distribution in dataset:")
        print("+-+" * 30)
        for gender, count in gender_counts.items():
            print(f"{gender}: {count}")

    drop_cols = [config.TARGET]
    if not config.WITH_GENDER:
        drop_cols.append("Sex")

    X = df.drop(columns=drop_cols)
    y = config.encode_target(df[config.TARGET])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLS,
            ),
            ("num", "passthrough", config.NUMERICAL_COLS),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train()
