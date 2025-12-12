from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model_name = "/mnt/workspace/model/Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
).to(device)

def generate_text(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        temperature=0.3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

instruction = """You are a medical diagnosis assistant. Predict the diagnosis from the patient data. Choose ONE from: [Bronchitis, Cold, Flu, Healthy, Pneumonia]. Output ONLY the disease name."""

prompts = [
    {'input': "21|Female|Runny nose|Headache|Sore throat|108|37.5|125/109|99"},
    {'input': "36|Male|Body ache|Cough|Fever|73|39.6|165/64|95"},
    {'input': "63|Female|Cough|Fever|Headache|105|37.1|122/63|93"}
]

for prompt in prompts:
    dialogue = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    output = generate_text(dialogue, max_new_tokens=10)
    diagnosis = output.strip().split()[0]
    print(f"Input: {prompt['input']}")
    print(f"Diagnosis: {diagnosis}\n")

