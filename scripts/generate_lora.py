import argparse
import sys
import time
import random
import string

import torch
import yaml
from peft import get_peft_config, load_peft_weights, PeftConfig

from hyper_llm_modulator.utils import (
    get_layers,
    embed_texts,
)
from hyper_llm_modulator.hyper_modulator import (
    HyperModulator,
    load_hypermod_checkpoint,
    save_lora,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns


def add_full_stop(s):
    s = s.strip()
    # check if s ends with . or .*
    if s[-1].isalpha():
        s += "."
    return s


def load_hypermod(hypermod_dir, device):
    checkpoint_path = f"{hypermod_dir}/hypermod.pt"
    (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(checkpoint_path, device)
    return (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LoRA for a given task description.",
    )
    parser.add_argument(
        "hypermod_dir",
        type=str,
        help="Directory containing hypermod checkpoint and configuration.",
    )
    parser.add_argument(
        "task_desc",
        type=str,
        nargs="?",
        default=None,
        help="Task description for which to generate LoRA.",
    )
    parser.add_argument(
        "--task_desc_file",
        type=str,
        help="File containing the task description.",
    )
    args = parser.parse_args()

    hypermod_dir = args.hypermod_dir

    if args.task_desc_file:
        with open(args.task_desc_file, "r") as f:
            task_desc = f.read().strip("\"' ")
    elif args.task_desc is None and not sys.stdin.isatty():
        task_desc = sys.stdin.read().strip("\"' ")
    else:
        task_desc = args.task_desc.strip("\"' ")

    print(f"\nGenerating LoRA for description:\n\n{task_desc}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load metadata
    args = argparse.Namespace(**yaml.safe_load(open(f"{hypermod_dir}/args.yaml", "r")))
    peft_config = get_peft_config(PeftConfig.from_json_file(f"{hypermod_dir}/adapter_config.json"))
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod(hypermod_dir, device)
    layer_indices = range(len(get_layers(model)))
    layer_indices = torch.tensor(layer_indices, dtype=torch.long, device=device)
    emb_size = emb_model.config.hidden_size

    # generate loras
    task_emb = embed_texts([task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device)
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
    lora_dir = f"{hypermod_dir}/extras/user_generated/{curtime}_{uuid}/"
    save_lora(lora_sd, peft_config, lora_dir)
    with open(f"{lora_dir}/task_desc.txt", "w") as f:
        f.write(task_desc)
    print(f"Saved lora to {lora_dir}")
