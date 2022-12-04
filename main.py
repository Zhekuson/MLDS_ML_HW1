import json

import joblib
import numpy as np
from fastapi import FastAPI, Request, Form, File, UploadFile
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

model = None
real_cols = ["seats", "year", "km_driven", "mileage", "engine", "max_power"]
cat_cols = ["fuel", "seller_type", "transmission", "owner"]
cols_to_save = real_cols + cat_cols
app = FastAPI()


class Item(BaseModel):
    name: str = None
    year: int = None
    selling_price: int = None
    km_driven: int = None
    fuel: str = None
    seller_type: str = None
    transmission: str = None
    owner: str = None
    mileage: str = None
    engine: str = None
    max_power: str = None
    torque: str = None
    seats: float = None


class Items(BaseModel):
    objects: List[Item]


@app.on_event("startup")
def on_start():
    try:
        global model
        model = joblib.load("best_pipeline.joblib")
    except Exception as e:
        print(e)
        exit(1)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    car = pd.DataFrame([x.dict() for x in [item]])
    car = car[cols_to_save]
    return float(model.predict(car))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    car = pd.DataFrame([x.dict() for x in items])
    car = car[cols_to_save]
    return list(model.predict(car))


@app.post("/predict_items")
async def predict_items_file(file: UploadFile = File()):
    file_type = file.content_type[file.content_type.find("/") + 1:]
    if file_type == "json":
        cars = pd.DataFrame([x.dict() for x in json.load(file.file)])
    elif file_type == "csv":
        cars = pd.read_csv(file)
    cars = cars[cols_to_save]
    cars["selling_price"] = model.predict(cars)
    return cars


@app.get('/')
async def root():
    return {"message": "working!"}


@app.get("/example")
def get_example():
    return Item(name="Maruti Swift Dzire VDI,2014",
                year=2014,
                torque="190Nm@ 2000rpm",
                selling_price=450000,
                km_driven=145500,
                fuel="Diesel",
                seller_type="Individual",
                transmission="Manual",
                owner="First Owner",
                mileage=23.40,
                engine=1248,
                max_power=74.00,
                seats=5)
