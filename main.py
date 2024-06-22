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


model=tf.keras.models.model_from_json('{"module": "keras", "class_name": "Sequential", "config": {"name": "sequential_2", "trainable": true, "dtype": "float32", "layers": [{"module": "keras.layers", "class_name": "InputLayer", "config": {"batch_shape": [null, 50], "dtype": "float32", "sparse": false, "name": "input_layer_2"}, "registered_name": null}, {"module": "keras.layers", "class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "dtype": "float32", "input_dim": 138137, "output_dim": 64, "embeddings_initializer": {"module": "keras.initializers", "class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "registered_name": null}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false}, "registered_name": null, "build_config": {"input_shape": [null, 50]}}, {"module": "keras.layers", "class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "zero_output_for_mask": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "recurrent_initializer": {"module": "keras.initializers", "class_name": "OrthogonalInitializer", "config": {"gain": 1.0, "seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "seed": null}, "registered_name": null, "build_config": {"input_shape": [null, 50, 64]}}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 32]}}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 10]}}], "build_input_shape": [null, 50]}, "registered_name": null, "build_config": {"input_shape": [null, 50]}, "compile_config": {"optimizer": {"module": "keras.optimizers", "class_name": "RMSprop", "config": {"name": "rmsprop", "learning_rate": 0.0010000000474974513, "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "loss_scale_factor": null, "gradient_accumulation_steps": null, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}, "registered_name": null}, "loss": {"module": "builtins", "class_name": "function", "config": "binary_crossentropy", "registered_name": "function"}, "loss_weights": null, "metrics": ["accuracy"], "weighted_metrics": null, "run_eagerly": false, "steps_per_execution": 1, "jit_compile": false}}').load_weights("politicpolarity.weights.h5")    
tk=tokenizer("tokenizer.txt")

@app.post("/predict/text")
def predict_text(body: PredModel):
    pred_data=np.array([tk.text_to_sequence(body.text)])
    #print(model.predict(pred_data)[0])
    return {"prediction": str(model.predict(pred_data)[0][0])}
