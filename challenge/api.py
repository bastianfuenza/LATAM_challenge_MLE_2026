import logging
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from challenge.logging_config import setup_logging
from challenge.model import DelayModel, InputDataException, ModelNotLoadedException

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
model = DelayModel()

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictRequest(BaseModel):
    flights: List[FlightData]

class PredictResponse(BaseModel):
    predict: List[int]

@app.get("/health", status_code=status.HTTP_200_OK)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        data_list = [flight.dict() for flight in request.flights]
        data = pd.DataFrame(data_list)

        features = model.preprocess(data)

        predictions = model.predict(features)

        return {"predict": predictions}
    except InputDataException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ModelNotLoadedException as e:
        # Service is up but can't serve predictions yet
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
