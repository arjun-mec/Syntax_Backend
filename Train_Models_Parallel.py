import pandas as pd
from prophet import Prophet
import pickle
from multiprocessing import Pool

df = pd.read_csv("ML_data.csv")


def train_model(args):
    state, district, category = args
    try:
        df_filtered = df[(df["State/UT"] == state) & (df["District"] == district)]
        df_category = df_filtered[df_filtered["Category"] == category].copy()
        if not df_category.empty:
            model = Prophet(changepoint_prior_scale=0.01)
            model.fit(df_category[["ds", "y"]])
            return (state, district, category), model
        else:
            return (state, district, category), None
    except Exception as e:
        print(f"Error training model for {state}, {district}, {category}: {e}")
        return (state, district, category), None


if __name__ == "__main__":
    models = {}
    tasks = []
    for state in df["State/UT"].unique():
        for district in df["District"].unique():
            for category in ["Outcome", "ECT", "IF", "SS", "DL", "GP"]:
                tasks.append((state, district, category))

    with Pool() as pool:
        results = pool.map(train_model, tasks)

    for key, model in results:
        if model is not None:
            models[key] = model

    with open("trained_models.pkl", "wb") as f:
        pickle.dump(models, f)
