import ktrain
from ktrain import text

# Function to perform data preprocessing using a ktrain transformer
def perform_data_preprocessing(transformer, X_train, y_train, X_test, y_test):
    """
    Perform data preprocessing using a ktrain transformer.

    Args:
    transformer (ktrain.text.Transformer): The ktrain transformer object.
    X_train (list): List of input training data.
    y_train (list): List of corresponding training labels.
    X_test (list): List of input test data.
    y_test (list): List of corresponding test labels.

    Returns:
    ktrain.text.preprocessor.TextPreprocessor: Preprocessed training data.
    ktrain.text.preprocessor.TextPreprocessor: Preprocessed test data.
    """
    # Preprocess the training data
    train = transformer.preprocess_train(X_train.to_list(), y_train.to_list())

    # Preprocess the test data
    val = transformer.preprocess_test(X_test.to_list(), y_test.to_list())

    return train, val
