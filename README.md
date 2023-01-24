# learning-pipeline-plugin


## Installation

```console
sudo apt-get update
sudo apt-get install -y python3-pip python3-pil

pip3 install poetry
poetry build
pip3 install dist/learning_pipeline_plugin-<PACKAGE_VERSION>.whl
```


## Usage

To collect data, create a pipe that inherits from `learning_pipeline_plugin.collect_pipe.CollectPipeBase`
and define `interpret_inputs()`.

Example:
```python
from typing import Optional
from learning_pipeline_plugin.collect_pipe import CollectPipeBase, DataDict

class CollectPipe(CollectPipeBase):
    def interpret_inputs(self, inputs) -> Optional[DataDict]:
        img, probs, feature = inputs
        return {
            "image": img,
            "feature_vector": feature,
            "other_data": {
                "probabilities": probs
            }
        }
```

`interpret_inputs()` gets the previous pipe output and must return `DataDict` or `None`.

`DataDict` is TypedDict for type hint, and must have following properties:

- `image`: PIL.Image
- `feature_vector`: vector with shape (N,)
- `other_data`: any data used for calculating uncertainty

Then, instantiate this and connect to other pipes:

```python
def main():
    [...]

    collect_pipe = CollectPipe(...)

    prev_pipe.connect(collect_pipe)
    collect_pipe.connect(after_pipe)
```
