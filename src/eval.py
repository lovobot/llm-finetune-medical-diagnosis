import json
import re
import torch
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score

def postprocess_response(response):
    valid_labels = {'bronchitis', 'cold', 'flu', 'healthy', 'pneumonia'}
    match = re.search(r'\b(' + '|'.join(valid_labels) + r')\b', response, re.IGNORECASE)
    return match.group().title() if match else 'Unknown'

def predict(instruction, input_text, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    )
    
    return postprocess_response(response)

def predict_testset(testset, model, tokenizer):
    
    results = []
    for idx, example in enumerate(tqdm(testset, desc="Evaluating", unit="sample")):
        result = {
            'input': example['input'],
            'true_label': example['output'],
            'predicted_label': None,
            'is_correct': False
        }
        
        try:
            pred = predict(
                example['instruction'],
                example['input'],
                model,
                tokenizer
            )
            result['predicted_label'] = pred
            result['is_correct'] = (pred == result['true_label'])
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            result['error'] = str(e)
        
        results.append(result)
    
    return results

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/Meta-Llama-3-8B-Instruct/",
        use_fast=False,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "./model/sft/checkpoint-4860",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    test_df = pd.read_json('./testset.json')
    testset = Dataset.from_pandas(test_df)
    
    detailed_results = predict_testset(testset, model, tokenizer)
    
    results_df = pd.DataFrame(detailed_results)
    
    valid_results = results_df[results_df['predicted_label'] != 'Unknown']
    
if not valid_results.empty:
    all_labels = ['Bronchitis', 'Cold', 'Flu', 'Healthy', 'Pneumonia']
    
    present_labels = list(set(valid_results['true_label']).union(set(valid_results['predicted_label'])))
    final_labels = [l for l in all_labels if l in present_labels]
    
    print("\nAggregated Metrics:")
    print(classification_report(
        valid_results['true_label'],
        valid_results['predicted_label'],
        labels=all_labels,
        target_names=all_labels,
        digits=4,
        zero_division=0
    ))
    
    print(f"Accuracy: {accuracy_score(valid_results['true_label'], valid_results['predicted_label']):.4f}")
else:
    print("\nNo valid predictions available for metrics calculation.")

if len(results_df) > 0:
    error_rate = 1 - len(valid_results)/len(results_df)
    print(f"\nError Analysis:\n- Total samples: {len(results_df)}")
    print(f"- Valid predictions: {len(valid_results)} ({1-error_rate:.2%})")
    print(f"- Invalid predictions: {len(results_df)-len(valid_results)} ({error_rate:.2%})")
else:
    print("\nNo test samples processed.")