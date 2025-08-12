from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from endpoint import predict
from pydantic import BaseModel, conlist

api_router = APIRouter()


class PredictionRequest(BaseModel):
    features: conlist(float, min_length=1, max_length=1)


class ModelResponse(BaseModel):
    prediction: int


@api_router.post("/predict", response_model=ModelResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        prediction = predict(request.features)
        # return prediction
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
