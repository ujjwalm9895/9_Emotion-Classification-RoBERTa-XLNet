import pandas as pd
import ktrain
from ktrain import text

# Create a class for the XLNet model
class XLNet:
    def __init__(self):
        """
        Initialize XLNet model attributes.

        Attributes:
        model_name (str): Name of the XLNet model (e.g., "xlnet-base-cased").
        maxlen (int): Maximum sequence length for tokenization.
        classes (list): List of class labels.
        batch_size (int): Batch size for data processing.
        """
        self.model_name = "xlnet-base-cased"
        self.maxlen = 128
        self.classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.batch_size = 32

    def create_transformer(self):
        """
        Create an XLNet transformer for text classification.

        Returns:
        ktrain.text.Transformer: XLNet transformer configured with specified attributes.
        """
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)
