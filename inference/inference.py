import keras
import numpy as np
from .sampling_strategies.sample_random import *

class Generative_inference:
    """ 
    This class facilitates text generation by utilizing a provided Keras model, 
    tokenizer, and search strategy. It allows for the generation of text based 
    on an initial prompt.

    Example:
        ```
        >>> inference = Generative_inference(model = model,
        >>>                          tokenizer = tokenizer,
        >>>                          search_strategy=random_sampling_strategy)
        >>> inference.generate("Hello World")
         ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  Hello WorldAr things sayingWhen ruby...
        ```
    
    """
    def __init__(self,
                 model,
                 tokenizer,
                 search_strategy=random_sampling_strategy,
                 prompt="Hello World",
                 input_len=64,
                 padding_token=0,
                 **kwargs
                 ):
        """
        Constructor for Generative_inference class.

        Args:
            model: A Keras model used for text generation.
            tokenizer: Tokenizer used to encode and decode text.
            search_strategy: Strategy used for searching tokens during generation. Default is `random_sampling_strategy`
            prompt (str): The initial prompt for text generation. Default is "Hello World".
            input_len (int): Length of the input tokens. Default is 64.
            padding_token (int): Token used for padding. Default is 0.
        """
        self.search_strategy = search_strategy
        self.kwargs = **kwargs
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.padding_token = padding_token
        self.input_len = input_len

    def generate(self,
                 prompt=None,
                 generate_limit=50,
                 **kwargs):
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The prompt for text generation. If not provided, uses the default prompt.
            generate_limit (int): Maximum number of tokens to generate. Default is 50.
            **kwargs: Additional keyword arguments to be passed to the search_strategy.

        Returns:
            str: Generated text.
        """

        if prompt is None:
            prompt = self.prompt
        
        prompt_tokens = self.tokenizer.tokenizer.encode_as_ids(prompt)

        input_prompt_token_len = len(prompt_tokens)

        if len(prompt_tokens) > self.input_len:
            prompt_tokens = prompt_tokens[:self.input_len]
        elif len(prompt_tokens) < self.input_len:
            prompt_tokens = [self.padding_token] * (self.input_len - len(prompt_tokens)) + prompt_tokens
        else:
            pass

        model_input = keras.ops.convert_to_tensor(prompt_tokens)
        model_input = keras.ops.reshape(model_input, (1, self.input_len))

        gen_len = 0
        while gen_len < generate_limit:

            gen_len += 1

            model_output = self.model(model_input)
            output_token = self.search_strategy(outputs=model_output, pos_num=-1, self.kwargs)
            model_input = keras.ops.convert_to_numpy(model_input)
            model_input = np.concatenate((model_input, [[output_token]]), -1)
            model_input = model_input[:, 1 :]
            # model_input = keras.ops.convert_to_tensor(model_input)
        
        model_input = keras.ops.reshape(model_input, (self.input_len,))
        model_input = keras.ops.convert_to_numpy(model_input)

        model_output = self.tokenizer.tokenizer.decode_ids(model_input.tolist())

        return model_output
