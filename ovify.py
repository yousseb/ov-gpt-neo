from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from openvino.runtime import serialize
from openvino.tools import mo
from pathlib import Path
from transformers.onnx import export, FeaturesManager


gpt_neo_pt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_neo_pt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


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
    return gen_text


def convert_to_ir(model: GPTNeoForCausalLM, tokenizer: GPT2Tokenizer, path: Path) -> str:
    # define path for saving onnx model
    onnx_path = Path(path / 'gpt-neo-1.3B.onnx')
    onnx_path.parent.mkdir(exist_ok=True)

    # get model onnx config function for output feature format casual-lm
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='causal-lm')

    # fill onnx config based on pytorch model config
    onnx_config = model_onnx_config(model.config)

    # convert model to onnx
    print(f'Exporting to ONNX')
    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

    print(f' onnx_inputs: {onnx_inputs}')
    print(f'onnx_outputs: {onnx_outputs}')

    # convert model to openvino
    ov_model = mo.convert_model(onnx_path, compress_to_fp16=True,
                                input='input_ids[1,1..128],attention_mask[1,1..128]')

    # serialize openvino model
    print(f'Exporting to OpenVino IR')
    serialize(ov_model, str(onnx_path.with_suffix('.xml')))


if __name__ == '__main__':
    #test_transformers_model(gpt_neo_pt_model, gpt_neo_pt_tokenizer)
    Path('model').mkdir(parents=True, exist_ok=True)
    convert_to_ir(gpt_neo_pt_model, gpt_neo_pt_tokenizer, Path('model'))
    print(f'Done')
