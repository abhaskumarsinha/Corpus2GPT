import keras
import numpy as np
from .sampling_strategies.sample_random import *


class Generative_inference:
    """ Generate sentences using different search strategies.

    Example:
    ```
        >>> inference = Generative_inference(model = model, tokenizer = tokenizer)
        >>> inference.generate("Hello World")
        '<unk> <unk> <unk> <unk> H e l l o W o r l d is one of the most common phrase used in ...'
    ```
    """
    def __init__(self, 
                 model,
                 search_strategy,
                 tokenizer=None, 
                 vocab=None, 
                 prompt=[],
                 input_len=64,
                 padding_token = 0
                 ):
        self.search_strategy = search_strategy
        self.model = model
        
        if tokenizer is None and vocab is None:
            raise Exception('Either a tokenizer from MultiLanguageTokenizer or vocab must be provided!')
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            self.int_to_token = {i : vocab[i] for i in range(len(vocab))}
            self.token_to_int = {int_to_token[i] : i for i in range(len(vocab))} 
        
        self.search_strategy = search_strategy

        self.prompt = prompt
        self.padding_token = padding_token
        self.input_len = input_len
    
    def generate(self, 
                 prompt=None,
                 generate_limit=50,
                 **kwargs):

      
        if prompt is None:
            prompt = self.prompt
        
        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize_sentences([prompt])
        else:
            tokens = prompt.split()
        
        prompt_tokens = []
        for token in prompt:
            if self.tokenizer is None:
                prompt_tokens.append(self.token_to_int[token])
            else:
                try:
                    prompt_tokens.append(self.tokenizer.vocab_dict[token])
                except:
                    pass
        input_prompt_token_len = len(prompt_tokens)
        
        if len(prompt_tokens) > self.input_len:
            prompt_tokens = prompt_tokens[ : self.input_len]
        elif len(prompt_tokens) < self.input_len:
            prompt_tokens = [self.padding_token]*(self.input_len - len(prompt_tokens)) + prompt_tokens
        else:
            pass
        

        model_input = keras.ops.convert_to_tensor(prompt_tokens)
        model_input = keras.ops.reshape(model_input, (1, self.input_len))

        gen_len = 0
        while gen_len < generate_limit:

            gen_len += 1

            model_output = self.model(model_input)
            output_token = self.search_strategy(outputs=model_output, pos_num=-1, **kwargs)
            model_input = keras.ops.convert_to_numpy(model_input)
            model_input = np.concatenate((model_input, [[output_token]]), -1)
            model_input = model_input[:, 1 :]
            #model_input = keras.ops.convert_to_tensor(model_input)

        model_input = keras.ops.convert_to_numpy(model_input)

        model_output = []
        for ids in model_input[0]:
            if self.tokenizer is None:
                model_output += [self.int_to_token[ids]]
            else:
                model_output += [tokenizer.inverse_vocab[ids]]
        

        if self.tokenizer is not None:
            model_output = "".join(tokenizer.decode_tokens(" ".join(model_output)))
        else:
            model_output = " ".join(model_output)
         
        return model_output
