#!/usr/bin/env python
# coding=utf-8

from argparse import ArgumentParser, SUPPRESS
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from openvino.runtime import serialize
from openvino.tools import mo
from pathlib import Path
from transformers.onnx import export, FeaturesManager
import logging as log
import subprocess
import sys
from optimum.intel.neural_compressor import INCModelForCausalLM

neo_model = 'gpt-neo-125M'
# neo_model = 'gpt-neo-1.3B'
# neo_model = 'gpt-neo-2.7B'


def test_transformers_model(model: GPTNeoForCausalLM, tokenizer: GPT2Tokenizer) -> str:
    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=128,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
    return gen_text


def compress_model() -> str:
    # https://github.com/huggingface/optimum-intel/tree/main/examples/neural_compressor/language-modeling#gpt-2gpt-and-causal-language-modeling
    compress_command = [f'python',
                        f'run_clm.py',
                        f'--model_name_or_path=EleutherAI/{neo_model}',
                        f'--dataset_name=wikitext',
                        f'--dataset_config_name=wikitext-2-raw-v1',
                        f'--apply_quantization=true',
                        f'--quantization_approach=aware_training',
                        f'--apply_pruning=true',
                        f'--target_sparsity=0.02',
                        f'--num_train_epochs=4',
                        f'--max_train_samples=100',
                        f'--do_train=true',
                        f'--do_eval=true',
                        f'--verify_loading=true',
                        f'--output_dir=compressed-{neo_model}']
    log.info(f"Compress command: `{compress_command}`")
    log.info(f"Compressing EleutherAI/{neo_model}... (This may take a while!)")
    subprocess.run(compress_command, shell=True)
    return f'compressed-{neo_model}'


def convert_to_ir(model: GPTNeoForCausalLM, tokenizer: GPT2Tokenizer, path: Path) -> str:
    # define path for saving onnx model
    onnx_path = Path(path / neo_model).with_suffix('.onnx')
    onnx_path.parent.mkdir(exist_ok=True)

    # get model onnx config function for output feature format casual-lm
    _gpt_neo_pt_model = GPTNeoForCausalLM.from_pretrained(f'EleutherAI/{neo_model}')

    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(_gpt_neo_pt_model, feature='causal-lm')

    # fill onnx config based on pytorch model config
    onnx_config = model_onnx_config(model.config)

    # # Built-in dummy dataset
    # dataset = Datasets('pytorch')['dummy'](shape=(1, 1, 128))
    # # Built-in calibration dataloader and evaluation dataloader for Quantization.
    # dataloader = DataLoader(framework='pytorch', dataset=dataset)
    #
    # config = PostTrainingQuantConfig()
    # q_model = fit(
    #     model,
    #     conf=config,
    #     calib_dataloader=dataloader,
    #     eval_dataloader=dataloader)

    # convert model to onnx
    log.info(f'Exporting to ONNX')
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

    log.info(f' onnx_inputs: {onnx_inputs}')
    log.info(f'onnx_outputs: {onnx_outputs}')

    # convert model to openvino
    ov_model = mo.convert_model(onnx_path, compress_to_fp16=True,
                                input='input_ids[1,1..128],attention_mask[1,1..128]')

    # serialize openvino model
    log.info(f'Exporting to OpenVino IR')
    serialize(ov_model, str(onnx_path.with_suffix('.xml')))


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-c", "--compress", help="Use Intel Neural Compressor to compress model",
                      action='store_true')
    args.add_argument("-t", "--test-transformer", help="Perform one step to test the model",
                      action='store_true')
    return parser


if __name__ == '__main__':
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    Path('model').mkdir(parents=True, exist_ok=True)
    gpt_neo_pt_model = GPTNeoForCausalLM.from_pretrained(f'EleutherAI/{neo_model}')
    gpt_neo_pt_tokenizer = GPT2Tokenizer.from_pretrained(f'EleutherAI/{neo_model}')

    args = build_argparser().parse_args()

    if args.compress:
        compress_model()
        compressed_model_path = f'quant-{neo_model}'
        gpt_neo_pt_model = INCModelForCausalLM.from_pretrained(f'./compressed-{neo_model}')
        gpt_neo_pt_tokenizer = GPT2Tokenizer.from_pretrained(f'./compressed-{neo_model}')

    if args.test_transformer:
        test_transformers_model(gpt_neo_pt_model, gpt_neo_pt_tokenizer)

    convert_to_ir(gpt_neo_pt_model, gpt_neo_pt_tokenizer, Path('model'))
    log.info(f'Done')
