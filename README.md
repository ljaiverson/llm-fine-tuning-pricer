# llm-fine-tuning-pricer

This repo contains the code for fine-tuning LLM to predict the price for Amazon items in a curated datset from Amazon reviews dataset. Shout out to Ed Donner and his dedication to the [LLM engineering course](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/?srsltid=AfmBOopPit1vBlOvCfsp9V4Qk9iFH-3uYEEYgSychfy89DmYpj9V6iWX) with amazing projects. This repo aims for a clean implementation related to the LLM fine-tuning portion of the class projects. The code for the entire class (mostly jupyter notebooks) can be found [here](https://github.com/ed-donner/llm_engineering).

![LLM Fine-tuning Pipeline](imgs/llm-finetune.png)

## Requirements
This implementation is base on `Python>=3.11.10` environment. To install the dependent packages for this implementation for `Python 3.11.10` environment, run
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```
If error occurs during the environment setup using these commands, more detailed setup instructions can be found [here](https://github.com/ed-donner/llm_engineering).

## Running experiments

First, run dataset curation, then test out (after training) different models by the commands below.

### dataset curation
```
python dataset_curation.py
```

### Featuren engineering and experiment on basic models
```
python feat_eng_basic_models.py
```

### Zero-shot and fine-tuning with LLM
```
python llm_zero-shot.py
```

### Full fine-tuning GPT-4o-mini
```
python FullFT_4o-mini.py
```

### QLoRA on Llama 3.1 8B
```
pip install -q datasets requests torch peft bitsandbytes transformers trl accelerate sentencepiece wandb matplotlib
python qlora_llama3.1.py
```

### Testing
testing a specific tuned model (change FINETUNED_MODEL and REVISION in test_huggingface_model.py)
```
python test_huggingface_model.py
```
