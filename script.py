import os
import csv
from sentence_transformers import SentenceTransformer, util
import requests

# input = ["biology.csv", "english.csv", "maths.csv", "physics.csv", "chemistry.csv"]
# input = ["test.csv"]
input = ["biology.csv"]

class Inference:
    prompt: str
    output: str
    llm_output: str
    similarity_score: float

    def __init__(self, prompt: str, output: str):
        self.prompt = prompt
        self.output = output
        self.llm_output = ""
        self.similarity_score = 0.0

    def inference(self):
        url = "http://localhost:8000/generate"
        data = {
            "prompt": f"Provide a concise and direct answer to the following NCERT-based question, staying within the scope of the question: {self.prompt}",
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            self.llm_output = result["answer"]
        else:
            print("Failed:", response.status_code, response.text)

    def evaluate(self):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embedding1 = model.encode(self.output, convert_to_tensor=True)
        embedding2 = model.encode(self.llm_output, convert_to_tensor=True)
        self.similarity_score = round(util.pytorch_cos_sim(embedding1, embedding2).item(), 2)

def write(filename: str, inference: Inference):
    outputfile = "out" + "/" + filename 

    if not os.path.exists(outputfile) or os.path.getsize(outputfile) == 0:
        with open(outputfile, mode='a', newline='') as csv_file:
            fieldnames = ['prompt', 'output', 'llm_output', 'similarity_score']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(outputfile, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['prompt', 'output', 'llm_output', 'similarity_score'])
        writer.writerow({
            'prompt': inference.prompt,
            'output': inference.output,
            'llm_output': inference.llm_output,
            'similarity_score': inference.similarity_score
        })

def read(filename: str):
    with open(file="input" + "/" + filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            inf = Inference(row['prompt'], row['output'])
            inf.inference()
            inf.evaluate()
            write(filename=filename, inference=inf)
            line_count += 1
        print(f'Processed {line_count} lines.')

if __name__ == "__main__":
    for i in input:
        read(i)
        print(f'Processed {i} file.')
    print("All files processed successfully.")
    print("Inference and evaluation completed.")

