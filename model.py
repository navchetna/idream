import openvino_genai as ov_genai

pipe_llama = ov_genai.LLMPipeline("Llama-3.2-3B-Instruct", "CPU")
pipe_qwen = ov_genai.LLMPipeline("Qwen2.5-3B-Instruct", "CPU")
pipe_phi = ov_genai.LLMPipeline("Phi-3.5-mini-instruct-fp16-ov", "CPU")

def generate_text(prompt: str):
    result_llama = pipe_llama.generate(prompt, max_new_tokens=100, do_sample=False)
    result_qwen = pipe_qwen.generate(prompt, max_new_tokens=100, do_sample=False)
    result_phi = pipe_phi.generate(prompt, max_new_tokens=100, do_sample=False)
    
    return {
        "answer_llama": result_llama,
        "answer_qwen": result_qwen,
        "answer_phi": result_phi
    }