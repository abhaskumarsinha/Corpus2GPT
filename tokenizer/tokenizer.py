import sentencepiece as spm
import os
from utils import C2GModelBase
import re

class MultiLanguageTokenizer(C2GModelBase):
    """
    A class for tokenizing multilingual text using the SentencePiece 
    algorithm.

    Attributes:
        tokenizer (spm.SentencePieceProcessor): SentencePiece
        tokenizer instance.

    Methods:
        __init__():
            Initializes the MultiLanguageTokenizer object and creates a SentencePiece tokenizer instance.

        train_tokenizer_from_file(file_location, model_prefix='tokenizer', vocab_size=90):
            Trains the tokenizer using the input file and loads the trained model.

        tokenize_sentences(sentences):
            Tokenizes a list of sentences.

        decode_tokens(token_lists):
            Decodes a list of tokenized sentences back to human-readable sentences.

        add_newlines_to_files(file_list, max_line_length):
            Scans a list of text files and adds newline characters to maintain line length.

        prepare_gpt_training_data(file_location):
            Prepares input-output pairs for training a GPT model using the specified text file.

        tokenize_file(file_location, output_file_location):
            Tokenizes the content of the specified text file and saves the tokenized content to another file.
  
    Examples:
        ```
        >>> tokenizer = MultiLanguageTokenizer()
        >>> tokenizer.train_tokenizer_from_file("data/multi_lang_corpus.txt")

        >>> sentences = [
        ...     "Hello, how are you?",     # English
        ...     "नमस्ते, आप कैसे हैं?",   # Hindi
        ...     "你好，你好吗？",              # Chinese
        ...     "Bonjour, comment ça va?"  # French
        ... ]

        >>> tokenized_sentences = tokenizer.tokenize_sentences(sentences)
        >>> print("Tokenized sentences:", tokenized_sentences)
        Tokenized sentences: [['▁Hello', ',', '▁how', '▁are', '▁you', '?'], ['▁नमस्ते', ',', '▁आप', '▁कैसे', '▁हैं', '?'], ['▁你好', '，', '你好', '吗', '？'], ['▁Bonjour', ',', '▁comment', '▁ça', '▁va', '?']]

        >>> decoded_sentences = tokenizer.decode_tokens(tokenized_sentences)
        >>> print("Decoded sentences:", decoded_sentences)
        Decoded sentences: ['Hello, how are you?', 'नमस्ते, आप कैसे हैं?', '你好，你好吗？', 'Bonjour, comment ça va?']
        ```
    Example for the use in GPT Model training:
        ```
        >>> tokenizer = MultiLanguageTokenizer()
        >>> tokenizer.train_tokenizer_from_file('./corpus.txt')
        >>> tokenizer.tokenize_file('./corpus.txt', './corpus_tokenized.txt')
        
        >>> file_list = ['./corpus_tokenized.txt']
        >>> tokenizer.add_newlines_to_files(file_list, 16)

        >>> data = tokenizer.prepare_gpt_training_data('./corput_tokenized.txt')
        >>> inputs, outputs = keras.ops.convert_to_tensor(data[0]), keras.ops.convert_to_tensor(data[1])
        ```
    """

    def __init__(self):
        """
        Initializes the MultiLanguageTokenizer object and creates a SentencePiece 
        tokenizer instance.
        """
        self.tokenizer = spm.SentencePieceProcessor()
        
    #@classmethod
    #def add_class_name(cls, _config):
    #    return {cls.__name__ : _config}
        

    def train_tokenizer_from_file(self, file_location, model_prefix='tokenizer', vocab_size=90, user_defined_symbols= '@@'):
        """
        Trains the tokenizer using the input file and loads the trained model.

        Args:
            file_location (str): Path to the text file used for training.
            model_prefix (str, optional): Prefix for the model files generated 
                                          during training. Default is 'tokenizer'.
            vocab_size (int, optional): Vocabulary size for the tokenizer.
                                        Default is 90.
        """
        if not hasattr(self, 'vocab'):
            spm.SentencePieceTrainer.train(input=file_location, model_prefix=model_prefix, vocab_size=vocab_size, user_defined_symbols = user_defined_symbols)

        model_path = model_prefix + '.model'
        self.tokenizer.load(model_path)
        sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

        self.vocab_dict = {token: i for i, token in enumerate(self.vocabs)}

        # Creating inverse vocab dictionary (integer to token)
        self.inverse_vocab = {i: token for token, i in self.vocab_dict.items()}

        self._config = {'model_path' : model_path,
                        'model_file' : model_path,
                        'vocabs' : self.vocabs,
                        'vocab_dict' : self.vocab_dict,
                        'inverse_vocab' : self.inverse_vocab}
        self._config = self.add_class_name(self._config)


    def tokenize_sentences(self, sentences):
        """
        Tokenizes a list of sentences.

        Args:
            sentences (list of str): List of sentences to be tokenized.

        Returns:
            list of lists of str: List of tokenized sentences, 
            where each sentence is represented as a list of tokens.
        """
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentence = self.tokenizer.encode(sentence, out_type=str)
            tokenized_sentences.append(tokenized_sentence)
        return tokenized_sentences

    def decode_tokens(self, token_lists):
        """
        Decodes a list of tokenized sentences back to human-readable sentences.

        Args:
            token_lists (list of lists of str): List of tokenized sentences, 
                                                where each sentence is represented 
                                                as a list of tokens.

        Returns:
            list of str: List of decoded sentences.
        """
        decoded_sentences = []
        for token_list in token_lists:
            decoded_sentence = self.tokenizer.decode(token_list)
            decoded_sentences.append(decoded_sentence)
        return decoded_sentences

    def add_newlines_to_files(self, file_list, max_words_per_line):
        """
        Scans a list of text files and adds newline characters to keep all lines shorter than a specified number of words.

        Args:
            file_list (list of str): List of file paths to be scanned.
            max_words_per_line (int): Maximum allowed number of words for each line.

        Example:
            >>> file_list = ['file1.txt', 'file2.txt']
            >>> max_words_per_line = 5
            >>> add_newlines_to_files(file_list, max_words_per_line)

        Returns:
            None
        """
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                content = " ".join(content).replace('\n', ' ')

            new_content = []
            words = content.split()
            new_content = [' '.join(words[i:i+max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(content)) #new_content, for testing purposes

    def prepare_gpt_training_data(self, file_location):
        """
        Prepares input-output pairs for training a GPT model 
        using the specified text file.

        Args:
            file_location (str): Path to the text file containing training data.

        Returns:
            list of tuple of str: List of tuples, where each tuple contains an 
            input sequence and its corresponding output sequence.

        Example usage:
            Suppose we have a text file 'training_data.txt' with the following content:

            ```
            Hello World, How are you?
            I hope you are doing fine.

            >>> tokenizer = MultiLanguageTokenizer()
            >>> tokenizer.train_tokenizer_from_file("training_data.txt")
            >>> training_data = tokenizer.prepare_gpt_training_data("training_data.txt")
            >>> for pair in training_data:
            ...     print(pair)

            Output:
            ('Hello World, How are', 'I hope you are doing fine.')
            ('World, How are you?', 'hope you are doing fine.')
            ```
        """
        with open(file_location, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        tokenized_sentences = self.tokenize_sentences(lines)
        
        input_pairs = []
        output_pairs = []
        for tokens in tokenized_sentences:
            input_tokens = tokens[1:-3]
            output_tokens = tokens[2:-2] 
            input_sentence = self.tokenizer.decode(input_tokens)
            output_sentence = self.tokenizer.decode(output_tokens)
            input_pairs.append([input_sentence])
            output_pairs.append([output_sentence])

        return input_pairs, output_pairs

    def tokenize_file(self, file_location, output_file_location):
        """
        Tokenizes the content of the specified text file and saves the tokenized content to another file.

        Args:
            file_location (str): Path to the text file to be tokenized.
            output_file_location (str): Path to the output file to save the tokenized content.

        Example usage:
            Suppose we have a text file 'example.txt' with the following content:

            This is an example sentence.
            Another sentence for testing.

            >>> tokenizer = MultiLanguageTokenizer()
            >>> tokenizer.train_tokenizer_from_file("example.txt")
            >>> tokenizer.tokenize_file("example.txt", "tokenized_example.txt")
        """
        with open(file_location, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        tokenized_content = self.tokenize_sentences(lines)

        with open(output_file_location, 'w', encoding='utf-8') as output_file:
            for sentence_tokens in tokenized_content:
                output_file.write(' '.join(sentence_tokens) + '\n')

        tokenized_content = self.tokenize_sentences(lines)


import os
import numpy as np
import sentencepiece as spm

class SPM_Tokenizer:
    """
    A class for tokenizing text data in multiple languages using SentencePiece.
    
    Attributes:
        vocab_model_file (str): The file path to the pre-trained SentencePiece vocabulary model file.
        vocab_size (int): The size of the vocabulary for training the tokenizer.
        corpus (str): The file path to the corpus used for training the tokenizer if no pre-trained vocabulary model is provided.
        model_prefix (str): The prefix for the output files generated during training if no pre-trained vocabulary model is provided.
        input_size (int): The maximum sequence length for tokenized sequences.
        model_type (str): The type of SentencePiece model to train, default is "unigram".
        tokenizer (spm.SentencePieceProcessor): The SentencePiece tokenizer object.
    
    Methods:
        load_file(file_path): Loads and tokenizes text data from a file.
        load_dataset(list_files): Loads and tokenizes text data from a list of files, yielding input-output pairs for training.

    Examples:
    ```
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
            vocab_model_file (str): The file path to the pre-trained SentencePiece vocabulary model file.
            vocab_size (int): The size of the vocabulary for training the tokenizer.
            corpus (str): The file path to the corpus used for training the tokenizer if no pre-trained vocabulary model is provided.
            model_prefix (str): The prefix for the output files generated during training if no pre-trained vocabulary model is provided.
            input_size (int): The maximum sequence length for tokenized sequences.
            model_type (str): The type of SentencePiece model to train, default is "unigram".
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
            file_path (str): The file path to the text file.
            
        Returns:
            list: A list of tokenized sequences.
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
            list_files (list): A list of file paths to text files.
            
        Yields:
            tuple: A tuple containing input and output sequences.
        """
        for file in list_files:
            content = self.load_file(file)
            X = [line[:-1] for line in content]
            Y = [line[1:] for line in content]
            yield X, Y

