import sentencepiece as spm
import os

class MultiLanguageTokenizer:
    """
    A class for tokenizing multilingual text using the SentencePiece 
    algorithm.

    Attributes:
        tokenizer (spm.SentencePieceProcessor): SentencePiece
        tokenizer instance.

    Methods:
        __init__():
            Initializes the MultiLanguageTokenizer object and 
            creates a SentencePiece tokenizer instance.

        `train_tokenizer_from_file(file_location, model_prefix='tokenizer', vocab_size=90):`
            Trains the tokenizer using the input file and loads the trained model.
            
            Args:
                file_location (str): Path to the text file used for training.
                model_prefix (str, optional): Prefix for the model files generated 
                                              during training. Default is 'tokenizer'.
                vocab_size (int, optional): Vocabulary size for the tokenizer. 
                                            Default is 90.

        tokenize_sentences(sentences):
            Tokenizes a list of sentences.
            
            Args:
                sentences (list of str): List of sentences to be tokenized.

            Returns:
                list of lists of str: List of tokenized sentences, where each 
                sentence is represented as a list of tokens.

        decode_tokens(token_lists):
            Decodes a list of tokenized sentences back to human-readable 
            sentences.
            
            Args:
                token_lists (list of lists of str): List of tokenized sentences, 
                                                    where each sentence is represented 
                                                    as a list of tokens.

            Returns:
                list of str: List of decoded sentences.
    
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
    """

    def __init__(self):
        """
        Initializes the MultiLanguageTokenizer object and creates a SentencePiece 
        tokenizer instance.
        """
        self.tokenizer = spm.SentencePieceProcessor()

    def train_tokenizer_from_file(self, file_location, model_prefix='tokenizer', vocab_size=90):
        """
        Trains the tokenizer using the input file and loads the trained model.

        Args:
            file_location (str): Path to the text file used for training.
            model_prefix (str, optional): Prefix for the model files generated 
                                          during training. Default is 'tokenizer'.
            vocab_size (int, optional): Vocabulary size for the tokenizer.
                                        Default is 90.
        """
        spm.SentencePieceTrainer.train(input=file_location, model_prefix=model_prefix, vocab_size=vocab_size)

        model_path = model_prefix + '.model'
        self.tokenizer.load(model_path)
        os.remove(model_path)  # Removing the temporary model file after loading

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
