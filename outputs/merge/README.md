---
library_name: peft
license: other
base_model: /mnt/workspace/model/Meta-Llama-3-8B-Instruct/
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

# sft

This model is a fine-tuned version of [/mnt/workspace/model/Meta-Llama-3-8B-Instruct/](https://huggingface.co//mnt/workspace/model/Meta-Llama-3-8B-Instruct/) on the trainset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0329

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.7288        | 0.3086 | 500  | 0.3266          |
| 0.6878        | 0.6173 | 1000 | 0.3585          |
| 0.0659        | 0.9259 | 1500 | 0.1930          |
| 0.2312        | 1.2346 | 2000 | 0.0747          |
| 0.3197        | 1.5432 | 2500 | 0.1021          |
| 0.0006        | 1.8519 | 3000 | 0.0439          |
| 0.0           | 2.1605 | 3500 | 0.0565          |
| 0.0           | 2.4691 | 4000 | 0.0369          |
| 0.0001        | 2.7778 | 4500 | 0.0331          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.48.3
- Pytorch 2.3.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0