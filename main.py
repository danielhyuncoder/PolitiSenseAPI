from fastapi import FastAPI
import tensorflow as tf
from Tokenizer import tokenizer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredModel(BaseModel):
    text: str


tk=tokenizer("tokenizer.txt")
model=tf.keras.models.load_model("politicpolarity.h5")
@app.post("/predict/text")
def predict_text(body: PredModel):
    pred_data=np.array([tk.text_to_sequence(body.text)])
    #print(model.predict(pred_data)[0])
    return {"prediction": str(model.predict(pred_data)[0][0])}
