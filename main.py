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

def getModel():
    m=tf.keras.models.Sequential()
    m.add(tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 64))
    m.add(tf.keras.layers.LSTM(32, activation="relu"))
    m.add(tf.keras.layers.Dense(10))
    m.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=['accuracy'])
    return m
model=getModel().load_weights("politicpolarity.h5")    
tk=tokenizer("tokenizer.txt")

@app.post("/predict/text")
def predict_text(body: PredModel):
    pred_data=np.array([tk.text_to_sequence(body.text)])
    #print(model.predict(pred_data)[0])
    return {"prediction": str(model.predict(pred_data)[0][0])}
