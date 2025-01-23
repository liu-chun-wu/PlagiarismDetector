from transformers import LongformerTokenizer, LongformerForSequenceClassification

class LongformerClassifier:
    def __init__(self, model_name='allenai/longformer-base-4096', num_labels=2, max_length=512):
        """
        Initializes the LongformerClassifier with a tokenizer and a model.
        
        Parameters:
            model_name (str): The name of the pre-trained Longformer model.
            num_labels (int): The number of labels for the classification task.
            max_length (int): The maximum length for tokenized sequences.
        """
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.max_length = max_length

    def preprocess_function(self, examples):
        """
        Preprocesses the input examples by tokenizing and padding them.
        
        Parameters:
            examples (dict): A dictionary containing the text data.
        
        Returns:
            dict: Tokenized and padded sequences.
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

    def preprocess_dataset(self, dataset):
        """
        Applies the preprocessing function to the entire dataset.
        
        Parameters:
            dataset: The dataset to be tokenized and preprocessed.
        
        Returns:
            Dataset: The tokenized dataset.
        """
        return dataset.map(self.preprocess_function, batched=True)