"""Index API router. at the momenet used as a sample"""

from fastapi import APIRouter, Depends
from typing import Union, Annotated

from app.services import auth as auth_service
from app.services import user as user_service
from app.routers.api import auth_middleware
from utils import get_logger
from app.services.mnist_service import *


logger = get_logger("api index")

router = APIRouter(
    prefix="/train_model",
    tags=["training models"],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)


@router.post("/mnist")
def train_model():
    print("Router Hit")

    accuracy = main_fun()
    return {"accuracy": accuracy}
