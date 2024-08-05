from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

app = FastAPI()


class RequestData(BaseModel):
    state: str
    district: str
    year: int


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("ML_data.csv")

# Optimization: Create a dictionary to store models for each location and category
models_cache = {}


@lru_cache(maxsize=None)
def get_prophet_model(state, district, category):
    df_filtered = df[(df["State/UT"] == state) & (df["District"] == district)]
    df_category = df_filtered[df_filtered["Category"] == category].copy()
    model = Prophet(changepoint_prior_scale=0.01)
    model.fit(df_category[["ds", "y"]])
    return model


def predict_category_scores(df_filtered, year):
    predictions = {}
    state = df_filtered["State/UT"].iloc[0]
    district = df_filtered["District"].iloc[0]
    future = pd.DataFrame({"ds": [pd.to_datetime(f"{year}-07-01")]})

    for category in ["Outcome", "ECT", "IF", "SS", "DL", "GP"]:
        try:
            model_key = (state, district, category)
            if model_key not in models_cache:
                models_cache[model_key] = get_prophet_model(state, district, category)

            model = models_cache[model_key]
            forecast = model.predict(future)
            predictions[category] = min(100, int(forecast["yhat"].values[0].round(2)))
        except Exception as e:
            print("Error in predict_category_score")
            predictions[category] = 0
            print(e)
    return predictions


@app.post("/predict/")
async def main(data: RequestData):
    state = data.state
    district = data.district
    year = data.year

    try:
        df_filtered = df[(df["State/UT"] == state) & (df["District"] == district)]

        if df_filtered.empty:
            return {"error": "Data not found for the given location and year."}

        predictions = {"State/UT": state, "District": district, "Year": year}
        predictions.update(predict_category_scores(df_filtered, year))
        predictions["Overall"] = sum(
            v for v in predictions.values() if isinstance(v, int)
        )

        return predictions

    except Exception as e:
        print(e)
        return {"error": "An error occurred during prediction."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="Syntax_Backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        timeout_keep_alive=600,
    )
