# learning-pipeline-plugin

Plugin for Actcast application.
This plugin provides a base Pipe class for selecting and collecting data.

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

## Notifier

By default, the information output by this plugin is logged as actlog through the Notifier instance.
What information is output (and in what format) can be designed by using custom notifier.

To customize it, define a custom notifier class inheriting from AbstractNotfier,
and define `notify()` which gets a message as str.
Then, instantiate and pass it to the CollectPipe constructor.

Example of introducing a message length limit:
```python
from datetime import datetime, timezone
import actfw_core
from learning_pipeline_plugin.notifier import AbstractNotfier

class CustomNotifier(AbstractNotfier):
    def notify(self, message: str):
        if len(message) > 128:
            message = message[:128] + " <truncated>"
        actfw_core.notify(
            [
                {
                    "info": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )

def main():
    [...]

    collect_pipe = CollectPipe(
        ...,
        notifier=CustomNotifier()
    )
```

