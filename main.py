# pip install accelerate
# pip install transformers
# pip install safetensors

# GPUを持っているならば
#pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"

model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto", offload_folder="/path/to/offload_folder")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = """USER: AIによって私達の暮らしはどのように変わりますか？
ASSISTANT: """

token_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(
    input_ids=token_ids.to(model.device),
    max_new_tokens=300,
    do_sample=True,
    temperature=0.8,
    streamer=streamer,
)
reak
