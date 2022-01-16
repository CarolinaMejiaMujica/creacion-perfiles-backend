from fastapi import APIRouter,Response
import pandas as pd
from starlette.responses import JSONResponse

information = APIRouter()

@information.get("/info/")
def cantidades():
    return JSONResponse(content={"cantidadTotal":5,"cantidadAnalisis":3})

 