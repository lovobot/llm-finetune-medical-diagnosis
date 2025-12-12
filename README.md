# LLM Fine-tuning on Medical Diagnosis Data

Medical diagnosis often relies on structured patient data. This project demonstrates how a fine-tuned LLM (Meta-LLaMA-3-8B-Instruct) can automate diagnosis.

*The base model was chosen because it's instruction-tuned, enabling the model to follow prompts accurately with minimal fine-tuning*.

## Project Overview

This project demonstrates an end-to-end workflow for:

- Data preprocessing: Cleaning, merging features, and creating JSON datasets
- LoRA fine-tuning with LLaMA Factory: Instruction-tuned adaptation of LLaMA-3-8B-Instruct
- Evaluation: Performance assessment using key metrics
- Deployment-ready merged adapter for inference

## Dataset

- Features: Age, Gender, Symptoms, Heart Rate, Body Temperature, Blood Pressure, Oxygen Saturation 
- Labels: `['Bronchitis', 'Cold', 'Flu', 'Healthy', 'Pneumonia']`

## Results

- Accuracy: 0.9900  
- F1-score (macro avg): 0.9916 

