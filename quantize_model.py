import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
torch._dynamo.disable()

def quantize_model(model_id, save_dir, bits=4, group_size=128):
    device = "cuda:0"
    print(f"Loading tokenizer for GPTQ")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading orca dataset")
    orca_data = load_dataset("Open-Orca/OpenOrca", split="train")
    
    print("Preparing examples for quantization")
    examples = []
    for i in range(500):
        prompt = orca_data[i]["question"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        inputs = {k: v for k, v in inputs.items()}
        examples.append(inputs)
    
    print(f"Loading model: {model_id}")

    gptq_config = GPTQConfig(bits=bits, group_size=group_size, dataset="c4", tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"":0},
        trust_remote_code=True,
        quantization_config=gptq_config
    )
    
    print(f"Saving model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":

    # BLOOM-3B
    quantize_model("bigscience/bloom-3b", "./quantized_bloom_3b", bits=8, group_size=128)

    # BLOOM-7.1B
    # quantize_model("bigscience/bloom-7b1", "./quantized_bloom_7b1", bits=8, group_size=128)

    # OPT-13B
    #quantize_model("facebook/opt-13b", "./quantized_opt_13b", bits=8, group_size=128)

