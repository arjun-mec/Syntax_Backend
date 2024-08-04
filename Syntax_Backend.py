from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware

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


def predict_category_score(df_filtered, category, future_year, predictions):
    try:
        df_category = df_filtered[df_filtered["Category"] == category].copy()

        model = Prophet(changepoint_prior_scale=0.01)
        model.fit(df_category[["ds", "y"]])

        future = pd.DataFrame({"ds": [pd.to_datetime(f"{future_year}-07-01")]})
        forecast = model.predict(future)

        if forecast["yhat"].values[0] > 100:
            predictions[category] = 100
        else:
            predictions[category] = int((forecast["yhat"].values[0]).round(2))

    except Exception as e:
        print("Error in predict_category_score")
        predictions[category] = 0
        print(e)


@app.post("/predict/")
async def main(data: RequestData):
    state = data.state
    district = data.district
    year = data.year

    try:
        df_filtered = df[(df["State/UT"] == state) & (df["District"] == district)]
        predictions = {"State/UT": state, "District": district, "Year": year}

        target_categories = [
            "Outcome",
            "ECT",
            "IF",
            "SS",
            "DL",
            "GP",
        ]

        overall = 0
        for category in target_categories:
            predict_category_score(df_filtered, category, year, predictions)
            overall += predictions[category]

        predictions["Overall"] = overall

        return predictions
    except Exception as e:
        print(e)
        return {"error": "Data not found for the given location and year."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="Syntax_Backend:app", host="127.0.0.1", port=8000, reload=True)
