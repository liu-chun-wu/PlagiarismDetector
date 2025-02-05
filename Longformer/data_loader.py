from datasets import load_dataset, DatasetDict

class DatasetHandler:
    def __init__(self, dataset_name, test_size=0.2):
        """
        Initialize the DatasetHandler with a dataset name and test split size.

        :param dataset_name: str, the name of the dataset to load.
        :param test_size: float, the proportion of the dataset to include in the validation split.
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.dataset = None

    def load_dataset(self):
        """
        Load the dataset using the given dataset name.
        """
        self.dataset = load_dataset(self.dataset_name)
        print("Dataset loaded successfully.")

    def split_dataset(self):
        """
        Split the dataset into training, validation, and test sets.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        train_test = self.dataset['train'].train_test_split(test_size=self.test_size)
        self.dataset = DatasetDict({
            'train': train_test['train'],
            'validation': train_test['test'],
            'test': self.dataset['test']
        })
        print("Dataset split into train, validation, and test sets.")

    def get_dataset(self):
        """
        Get the processed dataset.

        :return: DatasetDict, the processed dataset with train, validation, and test splits.
        """
        if self.dataset is None:
            self.load_dataset()
            self.split_dataset()

        return self.dataset
