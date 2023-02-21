"""
Based on https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gpt2_text_prediction_demo/python/gpt2_text_prediction_demo.py
     and https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/223-gpt2-text-prediction/223-gpt2-text-prediction.ipynb
On Windows, you also need: https://aka.ms/vs/17/release/vc_redist.x64.exe
"""

import logging as log
import time
from pathlib import Path
import numpy as np
from numba import njit, float32
from openvino.inference_engine import IECore
from openvino.runtime import Core, get_version, PartialShape, Dimension, Type
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPTNeoConfig:
    model: str = Path('model') / Path('gpt-neo-1.3B.xml')
    device: str = 'CPU'           # 'CPU', 'GPU', 'Auto', etc..
    dynamic_shape: bool = True    # False is broken for now.
    top_k: int = 200              # Number of tokens with the highest probability which will be kept for generation
    top_p: float = 0.9            # Maximum probability, tokens with such a probability and lower will be kept for generation
    max_sequence_length = 128     # When dynamic_shapes = False, use this maximum sequence length for stop iteration
    temperature: float = 0.9


class GPTNeo:
    def __init__(self, config: GPTNeoConfig):
        self.config = config

        # create tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.eos_token_id = self.tokenizer.eos_token_id
        log.debug('Tokenizer configured')

        log.info('OpenVINO Runtime build: {}'.format(get_version()))
        self.core = Core()

        Path('cl_cache').mkdir(parents=True, exist_ok=True)
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.core.set_property({'CACHE_DIR': Path('cache')})

        if config.device == 'CPU':
            ie = IECore()
            cpu_caps = ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name="CPU")
            log.info('Available CPU Optimizations: {}'.format(cpu_caps))
            if 'BF16' in cpu_caps:
                self.core.set_property({'ENFORCE_BF16': 'YES'})

        # read model
        log.info('Reading model {}'.format(config.model))
        self.model = self.core.read_model(config.model)

        self.input_tensor = self.model.inputs[0].any_name

        # validate model
        self._validate_model()

        if not config.dynamic_shape and (
                self.model.inputs[0].partial_shape.is_dynamic or self.model.inputs[0].shape[1] != config.max_seq_len):
            self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(config.max_seq_len)])})

        if config.dynamic_shape:
            # assign dynamic shapes to every input layer
            for input_layer in self.model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[0] = -1
                input_shape[1] = -1
                self.model.reshape({input_layer: input_shape})

        # load model to the device
        self.compiled_model = self.core.compile_model(self.model, config.device)
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()
        log.info('Model {} is loaded to {}'.format(config.model, config.device))

    def _validate_model(self):
        # check number inputs and outputs
        if len(self.model.inputs) != 2:
            raise RuntimeError('Expected model with single input, while provided {}'.format(
                len(self.model.inputs)))
        if len(self.model.outputs) != 1:
            raise RuntimeError('Expected model with single output, while provided {}'.format(
                len(self.model.outputs)))

    # this function converts text to tokens
    def _tokenize(self, text):
        """
        tokenize input text using GPT2 tokenizer

        Parameters:
          text, str - input text
        Returns:
          input_ids - np.array with input token ids
          attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
        """

        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

    # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
    @staticmethod
    @njit(cache=True, fastmath=True)  # Best implementation (VERY FAST)
    def _softmax(x):
        '''
        Performs the softmax activation on a given set of inputs
        Input: x (N,k) ndarray (N: no. of samples, k: no. of nodes)
        Returns:
        Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
        '''
        max_x = np.zeros((x.shape[0], 1), dtype=x.dtype)
        for i in range(x.shape[0]):
            max_x[i, 0] = np.max(x[i, :])
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1).reshape((-1, 1))  # Alternative of keepdims=True for Numba compatibility

    @staticmethod
    def _process_logits(cur_length, scores, eos_token_id, min_length=0):
        """
        reduce probability for padded indices

        Parameters:
          cur_length - current length of input sequence
          scores - model output logits
          eos_token_id - index of end of string token in model vocab
          min_length - minimum length for appling postprocessing
        """
        if cur_length < min_length:
            scores[:, eos_token_id] = -float("inf")
        return scores

    @staticmethod
    def _get_top_k_logits(scores, top_k):
        """
        perform top-k sampling

        Parameters:
          scores - model output logits
          top_k - number of elements with the highest probability to select
        """
        filter_value = -float(np.inf)
        top_k = min(max(top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        retval = np.ma.array(scores, mask=indices_to_remove,
                             fill_value=filter_value, dtype=float).filled()
        return retval

    @staticmethod
    def _get_top_p_logits(scores, top_p):
        filter_value = -float("Inf")
        sorted_indices = np.argsort(-scores)
        sorted_logits = -np.sort(-scores)
        cumulative_probs = np.cumsum(GPTNeo._softmax(sorted_logits), axis=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
        sorted_indices_to_remove[:, 0] = 0
        np.put_along_axis(sorted_indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)
        filtred_scores = np.ma.array(scores, mask=sorted_indices_to_remove, fill_value=filter_value).filled()
        return filtred_scores

    def _generate_sequence(self, input_ids, attention_mask, eos_token_id):
        """
        text prediction cycle.

        Parameters:
          input_ids: tokenized input ids for model
          attention_mask: attention mask for model
          eos_token_id: end of sequence index from vocab
        Returns:
          predicted token ids sequence
        """
        output_key = self.compiled_model.output(0)

        while True:
            cur_input_len = len(input_ids[0])
            if not self.config.dynamic_shape:
                pad_len = self.config.max_sequence_length - cur_input_len
                model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
                model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
            else:
                model_input_ids = input_ids
                model_input_attention_mask = attention_mask
            outputs = self.compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[
                output_key]
            next_token_logits = outputs[:, cur_input_len - 1, :]

            # pre-process distribution
            next_token_logits = next_token_logits / self.config.temperature

            next_token_scores = self._process_logits(cur_input_len, next_token_logits, eos_token_id)

            if self.config.top_p < 1.0:
                next_token_scores = self._get_top_p_logits(next_token_scores, self.config.top_p)

            if self.config.top_k > 0:
                next_token_scores = self._get_top_k_logits(next_token_scores, self.config.top_k)

            # get next token id
            probs = self._softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)

            # break the loop if max length or end of text token is reached
            if cur_input_len == self.config.max_sequence_length or next_tokens[0] == eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)
        return input_ids

    def infer(self, prompt: str) -> str:
        input_ids, attention_mask = self._tokenize(prompt)

        start = time.perf_counter()
        output_ids = self._generate_sequence(input_ids, attention_mask, eos_token_id=self.eos_token_id)
        end = time.perf_counter()
        output_text = ""
        # Convert IDs to words and make the sentence from it
        for i in output_ids[0]:
            output_text += self.tokenizer.convert_tokens_to_string(self.tokenizer._convert_id_to_token(i))

        log.debug(f"OUTPUT: {output_text}")
        log.info(f"Generation took {end - start:.3f} s")
        return f'{output_text}'


if __name__ == '__main__':
    import sys
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    config = GPTNeoConfig()
    gpt = GPTNeo(config)

    def prompts():
        while True:
            yield input('You: ')

    for prompt in prompts():
        response = gpt.infer(prompt)
        print(f'GPTNeo: {response}\n')
        print("-" * 70)
