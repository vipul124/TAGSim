# TAGSim: Topic-Informed Attention Guided Similarity
This is the official GitHub repository for the implementation of the paper "**TAGSim: Topic-Informed Attention Guided Similarity Metric for Language Agnostic Sentence Comparisons**". Please cite our paper if you use this code.

## Requirements
- torch>=2.1.2
- transformers>=4.36.2
- scikit-learn>=1.3.2
- tensorflow>=2.15.0
- scipy>=1.11.4
- numpy>=1.24.4

## Installation
You can install this package using the following command. Also, the mentioned requirements and dependencies are automatically installed using this.
```
pip3 install git+https://github.com/vipul124/TAGSim.git
```

## Dataset
The test datasets that we used for evaluation purposes are made available in the [dataset](/dataset) folder. Datasets used for training of [TAGSim-base](/TAGSim/models/TAGSim-base) model are not uploaded due to size constraints.

## Use
### How to train model using your own dataset?
To train your model using our approach, you'll need to choose the following parameters.
- `dataPath` - this is a necessary parameter. You need to mention the path of the JSON train file containing an array of $n \times 2 \times 2$ dimensions (A sample train file is provided [here]())
- `bertModel` (optional) - the name of the LLM model you want to use to generate embeddings. By default, we are using __bert-base-multilingual-cased__ model. For the complete list of available models, you can [refer this](https://huggingface.co/transformers/v3.3.1/pretrained_models.html)
- `tfModel` (optional) - architecture of the model in `tensorflow.keras.models` format ([sample](https://github.com/vipul124/TAGSim/blob/main/TAGSim/training.py#L12-L17))
- `simModel` (optional) - similarity function.
- `epochs` (optional) - number of epochs. Default - 10.
```python
from TAGSim import trainModel, saveModel

myModel = trainModel(dataPath=dataPath, bertModel=bertModel, tfModel=tfModel, simModel=simModel, epochs=epochs)
saveModel(myModel, "name_of_model")
```

### Testing your model
Here is the approach for evaluating the model.
- `dataPath` - this is a necessary parameter. You need to mention the path of the JSON test file (testing datasets are available [here](/dataset))
- `modelPath` - this is a necessary parameter. You need to mention the path of your model folder here (ex: `'models/TAGSim-base'`)
- `bertModel` (optional) - the name of the LLM model you want to use to generate embeddings. By default, we are using __bert-base-multilingual-cased__ model. For the complete list of available models, you can [refer this](https://huggingface.co/transformers/v3.3.1/pretrained_models.html)
```python
from TAGSim import testModel

testModel(dataPath=dataPath, modelPath=modelPath, bertModel=bertModel)
```

## License
This project is licensed under the terms of the `MIT License`.
