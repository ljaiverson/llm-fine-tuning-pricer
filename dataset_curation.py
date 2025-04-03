import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from utils.items import Item
from utils.loaders import ItemLoader
from collections import Counter, defaultdict
import numpy as np
import pickle

def setup_environment():
    """Load environment variables and set API keys."""
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

def login_to_huggingface():
    """Log in to HuggingFace using the token from environment variables."""
    hf_token = os.environ['HF_TOKEN']
    login(hf_token, add_to_git_credential=True)

def load_datasets(dataset_names):
    """Load datasets for the given list of dataset names."""
    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())
    return items

def balance_dataset(items):
    """Balance the dataset by price and category."""
    slots = defaultdict(list)
    for item in items:
        slots[round(item.price)].append(item)

    np.random.seed(42)
    random.seed(42)
    sample = []
    for i in range(1, 1000):
        slot = slots[i]
        if i >= 240:
            sample.extend(slot)
        elif len(slot) <= 1200:
            sample.extend(slot)
        else:
            weights = np.array([1 if item.category == 'Automotive' else 5 for item in slot])
            weights = weights / np.sum(weights)
            selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
            selected = [slot[i] for i in selected_indices]
            sample.extend(selected)
    return sample

def split_dataset(sample):
    """Split the dataset into training and test sets."""
    random.seed(42)
    random.shuffle(sample)
    train = sample[:400_000]
    test = sample[400_000:402_000]
    return train, test

def save_datasets(train, test):
    """Save the training and test datasets as pickle files."""
    with open('train.pkl', 'wb') as file:
        pickle.dump(train, file)
    with open('test.pkl', 'wb') as file:
        pickle.dump(test, file)

def push_to_huggingface(train, test):
    """Push the datasets to HuggingFace Hub."""
    train_prompts = [item.prompt for item in train]
    train_prices = [item.price for item in train]
    test_prompts = [item.test_prompt() for item in test]
    test_prices = [item.price for item in test]

    train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
    test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    HF_USER = "ljaiverson"  # Replace with your HuggingFace username
    DATASET_NAME = f"{HF_USER}/pricer-data"
    dataset.push_to_hub(DATASET_NAME, private=True)

def main():
    """Main function to curate the dataset."""
    setup_environment()
    login_to_huggingface()

    dataset_names = [
        "Automotive",
        "Electronics",
        "Office_Products",
        "Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Appliances",
        "Musical_Instruments",
    ]

    items = load_datasets(dataset_names)
    sample = balance_dataset(items)
    train, test = split_dataset(sample)
    save_datasets(train, test)
    push_to_huggingface(train, test)

if __name__ == "__main__":
    main()
