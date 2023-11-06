# pip install accelerate
# pip install transformers
# pip install safetensors

# GPUを持っているならば
#pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto", offload_folder="/path/to/offload_folder")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def llm(question):
    prompt = f"""USER:{question}\nASSISTANT: """
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        streamer=streamer,
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

txt = input("質問を入力してください:")
llm(txt)
