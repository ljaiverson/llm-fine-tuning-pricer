import os
import re
import json
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from utils.testing import Tester  # Import the Tester class for testing models

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')

# Initialize APIs
openai = OpenAI()
claude = Anthropic()

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

# GPT-4o-mini model prediction
def gpt_4o_mini(item):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# GPT-4o frontier model prediction
def gpt_4o_frontier(item):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages_for(item),
        seed=42,
        max_tokens=5
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# Claude 3.5 Sonnet model prediction
def claude_3_point_5_sonnet(item):
    messages = messages_for(item)
    system_message = messages[0]['content']
    messages = messages[1:]
    response = claude.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=5,
        system=system_message,
        messages=messages
    )
    reply = response.content[0].text
    return get_price(reply)

# Example usage
if __name__ == "__main__":
    with open('train.pkl', 'rb') as file:
        train = pickle.load(file)

    with open('test.pkl', 'rb') as file:
        test = pickle.load(file)
        
    print("GPT-4o-mini prediction:", gpt_4o_mini(test[0]))
    print("GPT-4o-frontier prediction:", gpt_4o_frontier(test[0]))
    print("Claude 3.5 Sonnet prediction:", claude_3_point_5_sonnet(test[0]))

    # Testing all models
    print("\nTesting GPT-4o-mini:")
    Tester.test(gpt_4o_mini, test)

    print("\nTesting GPT-4o-frontier:")
    Tester.test(gpt_4o_frontier, test)

    print("\nTesting Claude 3.5 Sonnet:")
    Tester.test(claude_3_point_5_sonnet, test)

