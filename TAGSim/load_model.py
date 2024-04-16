import os
import requests
import tensorflow as tf


# pre-checks for model folder
if not os.path.exists("models"):
    print("no models directory found, creating...")
    os.mkdir("models")

# setting default to TAGSim-base if nothing is specified
DEFAULT_modelPath = 'models/TAGSim-base'

# downloading TAGSim-base
if not os.path.exists(DEFAULT_modelPath):
    print("downloading TAGSim-base...")
    os.mkdir(DEFAULT_modelPath)
    response = requests.get("https://raw.githubusercontent.com/vipul124/TAGSim/main/TAGSim/models/TAGSim-base/model.json")
    with open(os.path.join(DEFAULT_modelPath, "model.json"), 'wb') as file:
        file.write(response.content)
    response = requests.get("https://raw.githubusercontent.com/vipul124/TAGSim/main/TAGSim/models/TAGSim-base/weights.h5")
    with open(os.path.join(DEFAULT_modelPath, "weights.h5"), 'wb') as file:
        file.write(response.content)


# main function
def loadModel(model=DEFAULT_modelPath):
    if not os.path.exists(os.path.join("models", model)):
        if not os.path.exists(model):
            print("operation failed: Enter a valid path or a valid model name that is stored in 'models' directory")
            return None
        else:
            if not os.path.exists(os.path.join(model, "model.json")):
                print(f"operation failed: No 'model.json' found in {model}")
                return None
            elif not os.path.exists(os.path.join(model, "weights.h5")):
                print(f"operation failed: No 'weights.h5' found in {model}")
                return None
            else: 
                tfModel = tf.keras.models.model_from_json(open(os.path.join(model, "model.json")).read())
                tfModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
                tfModel.load_weights(os.path.join(model, "weights.h5"))
                return tfModel
    else:
        if not os.path.exists(os.path.join("models", model, "model.json")):
                print(f"operation failed: No 'model.json' found in {model}")
                return None
        elif not os.path.exists(os.path.join("models", model, "weights.h5")):
            print(f"operation failed: No 'weights.h5' found in {model}")
            return None
        else: 
            tfModel = tf.keras.models.model_from_json(open(os.path.join("models", model, "model.json")).read())
            tfModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
            tfModel.load_weights(os.path.join("models", model, "weights.h5"))
            return tfModel