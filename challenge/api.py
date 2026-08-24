import logging
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from challenge.logging_config import setup_logging
from challenge.model import DelayModel, InputDataException, ModelNotLoadedException

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
model = DelayModel()

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int = Field(..., ge=DelayModel.MIN_MONTH, le=DelayModel.MAX_MONTH)

    @validator('OPERA')
    def known_opera(cls, value: str) -> str:
        if value not in DelayModel.OPERA_CATEGORIES:
            raise ValueError(
                f"unknown airline '{value}', expected one of {DelayModel.OPERA_CATEGORIES}"
                )

        return value

    @validator('TIPOVUELO')
    def known_tipovuelo(cls, value: str) -> str:
        if value not in DelayModel.TIPOVUELO_CATEGORIES:
            raise ValueError(
                f"unknown flight type '{value}', expected one of {DelayModel.TIPOVUELO_CATEGORIES}"
            )

        return value

class PredictRequest(BaseModel):
    flights: List[FlightData]

class PredictResponse(BaseModel):
    predict: List[int]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exception: RequestValidationError
) -> JSONResponse:
    """Custom exception handler to keep same format on input error.
    Answer schema violations with 400."""
    logger.warning(f"Rejected request to {request.url.path}: {exception.errors()}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": jsonable_encoder(exception.errors())}
    )

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
