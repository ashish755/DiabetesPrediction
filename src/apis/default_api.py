from fastapi import APIRouter, File, Query, UploadFile

from src.services.model_prediction import get_prediction
from src.services.model_training import DiabetesPrediction
from src.utils.create_file import create_file

router = APIRouter()


@router.post(
    "/modelTraining",
    responses={
        201: {"model": str, "description": "Model has been trained"}
    },
    tags=["Model Traning and Prediction"],
    summary="Train Decision tree model",
    response_model_by_alias=True,
)
def train_model(
    file_name: UploadFile = File(...)
):
    file_path = create_file(file_name)

    # train the diabetes prediction model
    diabetes_prediction = DiabetesPrediction()
    diabetes_prediction.train_model(file_path)

    return {"Model trained successfully"}


@router.post(
    "/modelPrediction",
    responses={
        201: {"model": str, "description": "Model Prediction"}
    },
    tags=["Model Traning and Prediction"],
    summary="Train Decision tree model",
    response_model_by_alias=True,
)
def predict_model(
    gender: int = Query(None, description="0-Female, 1-Male, 0-Other"),
    age: int = Query(None, description="Age of a person"),
    hypertension: int = Query(None, description="0-No, 1-Yes"),
    heart_disease: int = Query(None, description="0-No, 1-Yes"),
    smoking_history: int = Query(
        None, description="0-No Info, 1-current, 2-ever, 3-former, 4-never, 5-not current"),
    bmi: float = Query(None, description="Body mass index range from (10-50)"),
    HbA1_level: float = Query(
        None, description="Hemoglobin A1c range from (1-10)"),
    blood_glucose_level: float = Query(
        None, description="Blood glucose leve range from (50-200)"),
):
    predictions = get_prediction(gender, age, hypertension, heart_disease,
                                 smoking_history, bmi, HbA1_level, blood_glucose_level)

    if predictions == 0:
        return {"The patient does not have diabetes"}
    else:
        return {"The patient has diabetes"}
