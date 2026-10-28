import datasets
import transformers

# 加载一下自定义tokenzier
model_path = "saved/midtrain/midtrain-qwen-0.1B/"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

# 加载样例数据
raw_datasets = datasets.load_dataset(
    "json", data_files="data/sample.jsonl"
)

sample_data1 = raw_datasets["train"][0]["conversation"]
sample_data2 = raw_datasets["train"][1]["conversation"]
sample_data = sample_data1 + sample_data2

system_message = "You are a helpful assistant"
roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
IGNORE_TOKEN_ID = -100
context_length = 2048

im_start = tokenizer.bos_token_id
im_end = tokenizer.eos_token_id
nl_tokens = tokenizer("\n").input_ids
print("im_start:", im_start)
print("im_end:", im_end)
print("nl_tokens:", nl_tokens)

_system = tokenizer("system").input_ids + nl_tokens
_user = tokenizer("user").input_ids + nl_tokens
_assistant = tokenizer("assistant").input_ids + nl_tokens
print("_system:", _system)

messages = sample_data

# 拼接多轮对话的模版
input_id, target = [], []
system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
input_id += system
target += [IGNORE_TOKEN_ID] * len(system)
assert len(input_id) == len(target)

for j, sentence in enumerate(messages):
    role = roles[sentence["role"]]
    _input_id = tokenizer(role).input_ids + nl_tokens + \
        tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
    input_id += _input_id
    if role == "<|im_start|>user":
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
    elif role == "<|im_start|>assistant":
        _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
            _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
    else:
        raise Exception("Undefine Role!")
    target += _target

assert len(input_id) == len(target)

# 按最大截断
input_id = input_id[:context_length]
target = target[:context_length]
mask = [int(k != tokenizer.pad_token_id) for k in input_id]
print("input_id: ", input_id)
print("target: ", target)
print("mask: ", mask)
