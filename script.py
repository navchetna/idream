import os
import csv
from sentence_transformers import SentenceTransformer, util
import requests
from model import generate_text

# input = ["biology.csv", "english.csv", "maths.csv", "physics.csv", "chemistry.csv"]
# input = ["test.csv"]
input = ["biology.csv"]

class Inference:
    prompt: str
    output: str
    llm_output_llama: str
    llm_output_qwen: str
    llm_output_phi: str
    similarity_score_llama: float
    similarity_score_qwen: float
    similarity_score_phi: float

    def __init__(self, prompt: str, output: str):
        self.prompt = prompt
        self.output = output
        self.llm_output_llama = ""
        self.llm_output_qwen = ""
        self.llm_output_phi = ""
        self.similarity_score_llama = 0.0
        self.similarity_score_qwen = 0.0
        self.similarity_score_phi = 0.0

    def inference(self):
        generated_texts = generate_text(f"Provide a concise and direct answer to the following NCERT-based question: {self.prompt}")
        
        self.llm_output_llama = generated_texts["answer_llama"]
        self.llm_output_qwen = generated_texts["answer_qwen"]
        self.llm_output_phi = generated_texts["answer_phi"]

    def evaluate(self):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embedding1 = model.encode(self.output, convert_to_tensor=True)

        embedding2_llama = model.encode(self.llm_output_llama, convert_to_tensor=True)
        self.similarity_score_llama = round(util.pytorch_cos_sim(embedding1, embedding2_llama).item(), 2)

        embedding2_qwen = model.encode(self.llm_output_qwen, convert_to_tensor=True)
        self.similarity_score_qwen = round(util.pytorch_cos_sim(embedding1, embedding2_qwen).item(), 2)

        embedding2_phi = model.encode(self.llm_output_phi, convert_to_tensor=True)
        self.similarity_score_phi = round(util.pytorch_cos_sim(embedding1, embedding2_phi).item(), 2)

def write(filename: str, inference: Inference):
    outputfile = "out" + "/" + filename 

    if not os.path.exists(outputfile) or os.path.getsize(outputfile) == 0:
        with open(outputfile, mode='a', newline='') as csv_file:
            fieldnames = ['prompt', 'output', 'llm_output_llama', 'llm_output_qwen', 'llm_output_phi', 'similarity_score_llama', 'similarity_score_qwen', 'similarity_score_phi']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(outputfile, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['prompt', 'output', 'llm_output_llama', 'llm_output_qwen', 'llm_output_phi', 'similarity_score_llama', 'similarity_score_qwen', 'similarity_score_phi'])
        writer.writerow({
            'prompt': inference.prompt,
            'output': inference.output,
            'llm_output_llama': inference.llm_output_llama,
            'llm_output_qwen': inference.llm_output_qwen,
            'llm_output_phi': inference.llm_output_phi,
            'similarity_score_llama': inference.similarity_score_llama,
            'similarity_score_qwen': inference.similarity_score_qwen,
            'similarity_score_phi': inference.similarity_score_phi
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

