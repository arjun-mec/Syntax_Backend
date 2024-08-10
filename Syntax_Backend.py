from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

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

# Load the trained models
with open("trained_models.pkl", "rb") as f:
    models_cache = pickle.load(f)


def predict_category_scores(state, district, year):
    predictions = {}
    future = pd.DataFrame({"ds": [pd.to_datetime(f"{year}-07-01")]})

    for category in ["Outcome", "ECT", "IF", "SS", "DL", "GP"]:
        try:
            model_key = (state, district, category)
            model = models_cache.get(model_key)

            if model:
                forecast = model.predict(future)
                predictions[category] = min(
                    100, int(forecast["yhat"].values[0].round(2))
                )
            else:
                predictions[category] = 0
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
        predictions = {"State/UT": state, "District": district, "Year": year}
        predictions.update(predict_category_scores(state, district, year))
        predictions["Overall"] = (
            predictions["Outcome"]
            + predictions["DL"]
            + predictions["ECT"]
            + predictions["GP"]
            + predictions["IF"]
            + predictions["SS"]
        )

        return predictions

    except Exception as e:
        print(e)
        return {"error": "An error occurred during prediction."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="Syntax_Backend:app", host="127.0.0.1", port=8000, reload=True)
