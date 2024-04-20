import sentencepiece as spm
import os
import re
import numpy as np

class SPM_Tokenizer:
    """
    A class for tokenizing text data in multiple languages using SentencePiece.
    
    Attributes:
    - vocab_model_file (str): The file path to the pre-trained SentencePiece vocabulary model file.
    - vocab_size (int): The size of the vocabulary for training the tokenizer.
    - corpus (str): The file path to the corpus used for training the tokenizer if no pre-trained vocabulary model is provided.
    - model_prefix (str): The prefix for the output files generated during training if no pre-trained vocabulary model is provided.
    - input_size (int): The maximum sequence length for tokenized sequences.
    - model_type (str): The type of SentencePiece model to train, default is "unigram".
    - tokenizer (spm.SentencePieceProcessor): The SentencePiece tokenizer object.
    
    Methods:
    - load_file(file_path): Loads and tokenizes text data from a file.
    - load_dataset(list_files): Loads and tokenizes text data from a list of files, yielding input-output pairs for training.

    Examples:
    ```python
        >>> # Create a new Tokenizer
        >>> SPM_Tokenizer(vocab_size = 5000, corpus='./stories.txt', input_size=100+1) # Context-width of GPT+1
        >>> tokenizer = SPM_Tokenizer(vocab_model_file='./tokenizer_.model')
        Success!
        >>> tokenizer.tokenizer.encode_as_ids(['Hello World', 'How are you?'])
        [[3063, 215, 920, 129, 1323], [654, 54, 217, 78]]
        >>> dataset = tokenizer.load_dataset(['./stories.txt'])
        >>> for (X, Y) in dataset:
        >>>     X=np.array(X)[0]
        >>>     Y=np.array(Y)[0]
        >>> tokenizer.tokenizer.decode_ids(X.tolist()), tokenizer.tokenizer.decode_ids(Y.tolist())
        ('The Project Gutenberg EBook of The Thousand and One Nights, Vol. I., by Anonymous This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the',
         'Project Gutenberg EBook of The Thousand and One Nights, Vol. I., by Anonymous This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms')
        
    ```
    """
    
    def __init__(self, vocab_model_file=None, vocab_size=5000, corpus=None, model_prefix='tokenizer_', input_size=65, model_type="unigram"):
        """
        Initializes the MultiLanguageTokenizer object.
        
        Parameters:
        - vocab_model_file (str): The file path to the pre-trained SentencePiece vocabulary model file.
        - vocab_size (int): The size of the vocabulary for training the tokenizer.
        - corpus (str): The file path to the corpus used for training the tokenizer if no pre-trained vocabulary model is provided.
        - model_prefix (str): The prefix for the output files generated during training if no pre-trained vocabulary model is provided.
        - input_size (int): The maximum sequence length for tokenized sequences.
        - model_type (str): The type of SentencePiece model to train, default is "unigram".
        """
        self.input_size = input_size
        if vocab_model_file is not None and os.path.exists(vocab_model_file):
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(vocab_model_file)
        else:
            if corpus is None:
                raise Exception('A corpus to train the tokenizer must be provided!')

            self.tokenizer = spm.SentencePieceTrainer.train(input=corpus, model_prefix=model_prefix, vocab_size=vocab_size, model_type=model_type)
        
        if self.tokenizer is not None:
            print('Success!')
    
    def load_file(self, file_path):
        """
        Loads and tokenizes text data from a file.
        
        Parameters:
        - file_path (str): The file path to the text file.
            
        Returns:
        - list: A list of tokenized sequences.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
            content = " ".join(content).replace('\n', ' ')
            content = self.tokenizer.encode_as_ids(content)
            new_content = [content[i:i+self.input_size] for i in range(0, len(content), self.input_size)]
            num_zeros = self.input_size - len(new_content[-1])
            padded_list = [0] * num_zeros + new_content[-1]
            new_content[-1] = padded_list
        return new_content
    
    def load_dataset(self, list_files):
        """
        Loads and tokenizes text data from a list of files, yielding input-output pairs for training.
        
        Parameters:
        - list_files (list): A list of file paths to text files.
            
        Yields:
        - tuple: A tuple containing input and output sequences.
        """
        for file in list_files:
            content = self.load_file(file)
            X = [line[:-1] for line in content]
            Y = [line[1:] for line in content]
            yield X, Y

