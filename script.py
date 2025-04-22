import csv
import openvino_genai as ov_genai
from sklearn.metrics import accuracy_score

home = "/home/thebeginner86/code/idream"

model_path = "TinyLlama"
pipe = ov_genai.LLMPipeline("/home/thebeginner86/code/idream/Llama-3.2-3B-Instruct", "CPU")

# input = ["biology.csv", "english.csv", "maths.csv", "physics.csv", "chemistry.csv"]
input = ["test.csv"]

class Inference:
    prompt: str
    output: str
    expected_output: str
    accuracy: float

    def __init__(self, prompt: str, output: str):
        self.prompt = prompt
        self.output = output
        self.expected_output = ""
        self.accuracy = 0.0

    def inference(self):
        self.expected_output = pipe.generate(self.prompt, max_new_tokens=100, do_sample=False)

    # here make req to Groq that would some model hosted
    # ask, evaluate this output with expected output
    def evaluate(self):
        pass


# logic it appends to the file
# avc
# cde
# efs
def write(filename: str, inference: Inference):
    outputfile = home + "/" + "out" + "/" + filename 
    with open(outputfile, mode='w') as csv_file:
        fieldnames = ['prompt', 'output', 'expected_output', 'accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'prompt': inference.prompt,
            'output': inference.output,
            'expected_output': inference.expected_output,
            'accuracy': inference.accuracy
        })

def read(filename: str):
    with open(file=home + "/" + "input" + "/" + filename, mode='r') as csv_file:
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

