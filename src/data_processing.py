import pandas as pd
import json
from sklearn.model_selection import train_test_split

def load_and_clean_data(input_path, columns_to_extract):

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {input_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

    required_cols = list(set(columns_to_extract))
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[columns_to_extract].drop_duplicates()

    return df_clean.dropna()


def concat_columns(df, input_cols, text):

    df[text] = df[input_cols].apply(lambda row: '|'.join(row.astype(str)), axis=1)
    df = df.drop(columns=input_cols)

    print(f"Created merged text column: {text}")    

    return df


def split_data(df, test_size=0.1, random_state=42):

    trainset, testset = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Diagnosis'],
        random_state=random_state
    )
    print(f"Split complete: Train({len(trainset)}), Test({len(testset)})")

    return trainset, testset


def create_json_data(trainset, testset, output_path_train, output_path_test):
    
    instruction = """You are a medical diagnosis assistant. Predict the diagnosis from the patient data. Choose ONE from: [Bronchitis, Cold, Flu, Healthy, Pneumonia]. Output ONLY the disease name."""
    
    def format_record(row):
        return {
            "instruction": instruction,
            "input": row['text'],
            "output": row['Diagnosis']
        }

    def save_dataset(df, output_path):
        data = [format_record(row) for _, row in df.iterrows()]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} records to {output_path}")

    save_dataset(trainset, output_path_train)
    save_dataset(testset, output_path_test)


if __name__ == '__main__':

    CONFIG = {
        "input_path": "./disease_diagnosis.csv",
        "columns": [
            'Age', 'Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Heart_Rate_bpm', 'Body_Temperature_C', 'Blood_Pressure_mmHg', 'Oxygen_Saturation_%', 'Diagnosis'
        ],
        "output_paths": {
            "train": "./trainset.json",
            "test": "./testset.json"
        }
    }

    try:
        df = load_and_clean_data(CONFIG["input_path"], CONFIG["columns"])
        df = concat_columns(df, CONFIG["columns"][:-1], text="text")
        trainset, testset = split_data(df)
        create_json_data(trainset, testset, CONFIG["output_paths"]["train"], CONFIG["output_paths"]["test"])

        print("Data processing completed successfully")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
