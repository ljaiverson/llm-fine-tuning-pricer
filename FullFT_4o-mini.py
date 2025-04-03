import os
import re
import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from utils.testing import Tester  # Import the Tester class for testing models
import wandb  # Import wandb for tracking fine-tuning progress

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')

# Initialize APIs
openai = OpenAI()

# Utility function to extract price from a string
def get_price(s):
    s = s.replace('$', '').replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0

# Generate messages for a given item
def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]

# Prepare data for fine-tuning in JSONL format
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str + '}\n'
    return result.strip()

# Write JSONL data to a file
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

# Upload JSONL file to OpenAI for fine-tuning
def upload_to_openai(filename, purpose="fine-tune"):
    with open(filename, "rb") as f:
        return openai.files.create(file=f, purpose=purpose)

# Fine-tune a model
def fine_tune_model(training_file_id, validation_file_id, model="gpt-4o-mini-2024-07-18", n_epochs=1, suffix="pricer"):
    return openai.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        seed=42,
        hyperparameters={"n_epochs": n_epochs},
        suffix=suffix
    )

# Fine-tune a model with wandb integration
def fine_tune_model_with_wandb(training_file_id, validation_file_id, model="gpt-4o-mini-2024-07-18", n_epochs=1, suffix="pricer"):
    wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}
    return openai.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        seed=42,
        hyperparameters={"n_epochs": n_epochs},
        integrations=[wandb_integration],
        suffix=suffix
    )

# Retrieve fine-tuned model name
def get_fine_tuned_model_name(job_id):
    return openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

# Fine-tuned model prediction
def gpt_fine_tuned(item, fine_tuned_model_name):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=messages_for(item),
        seed=42,
        max_tokens=7
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# Example usage
if __name__ == "__main__":
    with open('train.pkl', 'rb') as file:
        train = pickle.load(file)

    with open('test.pkl', 'rb') as file:
        test = pickle.load(file)

    # Prepare fine-tuning data
    fine_tune_train = train[:200]
    fine_tune_validation = train[200:250]
    write_jsonl(fine_tune_train, "fine_tune_train.jsonl")
    write_jsonl(fine_tune_validation, "fine_tune_validation.jsonl")

    # Upload files to OpenAI
    train_file = upload_to_openai("fine_tune_train.jsonl")
    validation_file = upload_to_openai("fine_tune_validation.jsonl")

    # Fine-tune the model with wandb integration
    fine_tune_job = fine_tune_model_with_wandb(train_file.id, validation_file.id)
    fine_tuned_model_name = get_fine_tuned_model_name(fine_tune_job.id)

    # Test the fine-tuned model
    print("Fine-tuned model prediction:", gpt_fine_tuned(test[0], fine_tuned_model_name))

    print("\nTesting Fine-tuned model:")
    Tester.test(lambda item: gpt_fine_tuned(item, fine_tuned_model_name), test)
