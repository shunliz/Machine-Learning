# Transformers

transformer是一个提供了所有你需要做预训练的模型推理或者训练的库。主要功能包括：

## Inference

### pipeline

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

### text generation

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json", push_to_hub=True)

generation_config = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

### cache

|       Cache Type       | Memory Efficient | Supports torch.compile() | Initialization Recommended | Latency | Long Context Generation |
| :--------------------: | :--------------: | :----------------------: | :------------------------: | :-----: | :---------------------: |
|     Dynamic Cache      |        No        |            No            |             No             |   Mid   |           No            |
|      Static Cache      |        No        |           Yes            |            Yes             |  High   |           No            |
|    Offloaded Cache     |       Yes        |            No            |             No             |   Low   |           Yes           |
| Offloaded Static Cache |        No        |           Yes            |            Yes             |  High   |           Yes           |
|    Quantized Cache     |       Yes        |            No            |             No             |   Low   |           Yes           |
|  Sliding Window Cache  |        No        |           Yes            |            Yes             |  High   |           No            |

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

past_key_values = DynamicCache()
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)
```



## Training

### Trainer

[Trainer](https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/trainer#transformers.Trainer) 包含了模型训练的所有步骤.

1. 计算一个训练步的损失
2. 反向传递计算梯度
3. 根据梯度更新参数
4. 直到达到训练epoch数。

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Accelerate

加几行代码到Pytorch代码就可以实现模型并行加速

```python
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```

DistributedDataParallel

```
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #change rank as per the node
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

FullyShardedDataParallel

```
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

运行

```
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

### PEFT

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

# create LoRA configuration object
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)

model.add_adapter(lora_config, adapter_name="lora_1")
trainer = Trainer(model=model, ...)
trainer.train()
```

## quantization

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    device_map="auto",
    quantization_config=quantization_config, 
    torch_dtype="auto"
)
model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

