import random
import itertools

class LanguagePairDataset:
    """`LanguagePairDataset` is a synthetic data generation class that creates unique 
    vocabularies for different languages, establishes one-to-one word mappings between 
    languages, and generates paired language sentences. It supports dataset creation for 
    translation tasks, with functionality for appending to existing datasets or creating 
    new ones.

    Example:
        ```
        # Example usage:
        dataset = LanguagePairDataset()

        # Create vocab for the first and second languages
        dataset.create_vocabs(10, 3)  # First language
        dataset.create_vocabs(10, 3)  # Second language

        # Create a map between the first and second languages
        map_dict, reverse_dict = dataset.create_map(0, 1)

        # Write 5 sentence pairs to a file (append mode: True)
        dataset.write_dataset_to_file(map_dict, reverse_dict, max_len=3, n=5, file_path='dataset.txt', append=False)

        # Overwrite file with new 5 pairs (append mode: False)
        dataset.write_dataset_to_file(map_dict, reverse_dict, max_len=3, n=5, file_path='dataset.txt', append=True)

        print("Dataset written to 'dataset.txt'")
        ```
    """
    def __init__(self):
        # Valid characters to use for creating vocabularies
        self.valid_chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        # List to store all languages' vocabularies
        self.languages = []
        # Set to track all unique words used so far
        self.used_words = set()

    def create_vocabs(self, vocab_size, max_char):
        """
        Creates a vocabulary list with unique words of length `max_char`
        and appends it to the languages list.

        vocab_size: The number of words to generate for the new vocabulary.
        max_char: The maximum number of characters in each word.
        """
        new_vocab = []
        # Generate all possible combinations of characters of the specified length
        possible_combinations = list(itertools.product(self.valid_chars, repeat=max_char))
        
        # Shuffle combinations to ensure randomness
        random.shuffle(possible_combinations)
        
        # Create unique words until we hit the vocab size or run out of combinations
        for combination in possible_combinations:
            word = ''.join(combination)
            if word not in self.used_words:  # Ensure no duplicates across languages
                new_vocab.append(word)
                self.used_words.add(word)
            if len(new_vocab) == vocab_size:
                break
        
        # Save the new vocabulary as a new language
        if len(new_vocab) < vocab_size:
            print(f"Warning: Only {len(new_vocab)} unique words could be created.")
        self.languages.append(new_vocab)

    def create_map(self, lang1_idx, lang2_idx):
        """
        Creates a mapping between two languages' vocabularies.

        lang1_idx: Index of the first language in the languages list.
        lang2_idx: Index of the second language in the languages list.

        Returns:
            A tuple containing:
                - map_dict: A dictionary mapping each word of the first language to a word in the second language.
                - reverse_dict: A dictionary mapping each word of the second language back to the first language.
        """
        # Retrieve vocabularies of both languages
        lang1 = self.languages[lang1_idx]
        lang2 = self.languages[lang2_idx]

        if len(lang1) != len(lang2):
            raise ValueError("Both languages must have the same number of words for one-to-one mapping.")
        
        # Create the forward mapping
        map_dict = {lang1[i]: lang2[i] for i in range(len(lang1))}
        # Create the reverse mapping
        reverse_dict = {lang2[i]: lang1[i] for i in range(len(lang1))}
        
        return map_dict, reverse_dict

    def create_translate_language_instance(self, map_dict, reverse_dict, max_len):
        """
        Creates a random sentence from the first language and its corresponding translation.

        map_dict: Dictionary that maps words from the first language to the second.
        reverse_dict: Dictionary that maps words from the second language back to the first.
        max_len: Maximum number of words in the sentence.
        
        Returns:
            A tuple containing:
                - Sentence in the first language (concatenated string of words).
                - Corresponding sentence in the second language (concatenated string of translated words).
                - Original sentence in the first language (concatenated string of original words).
        """
        # Select random words from the first language (keys of map_dict)
        first_language_words = random.sample(list(map_dict.keys()), max_len)
        
        # Get the corresponding words in the second language
        second_language_words = [map_dict[word] for word in first_language_words]
        
        # Create concatenated strings of sentences
        first_sentence = ' '.join(first_language_words)
        second_sentence = ' '.join(second_language_words)
        
        # Return the sentences as concatenated strings
        return first_sentence, second_sentence, first_sentence

    def write_dataset_to_file(self, map_dict, reverse_dict, max_len, n, file_path, append=True, instance_split=None, sentence_split=None):
        """
        Writes `n` sentence pairs to a file. Each pair contains a sentence in the first language
        and its corresponding translation in the second language.

        map_dict: Dictionary mapping words from the first language to the second language.
        reverse_dict: Dictionary mapping words from the second language to the first language.
        max_len: Maximum number of words in each sentence.
        n: Number of sentence pairs to generate.
        file_path: The file path to write the dataset to.
        append: Boolean flag to append to the file if it exists, otherwise overwrite.
        instance_split: Optional character to separate sentence pairs in the file.
        sentence_split: Optional character to separate two sentences in the file.
        """
        # Open the file in append mode if append=True, otherwise overwrite
        mode = 'a' if append else 'w'
        
        with open(file_path, mode) as f:
            for _ in range(n):
                first_sentence, second_sentence, _ = self.create_translate_language_instance(map_dict, reverse_dict, max_len)
                # Write the first language sentence
                f.write(first_sentence + '\n')
                # If a sentence split char is provided, write it as a separator
                if sentence_split:
                    f.write(sentence_split + '\n')
                # Write the corresponding second language sentence
                f.write(second_sentence + '\n')
                # If a special character is provided, write it as a separator
                if instance_split:
                    f.write(instance_split + '\n')

    def test_translation_accuracy(self, map_dict, reverse_dict, first_sentence, translated_sentence, interrupt_errors = True):
        """
        Tests the translation accuracy by comparing how many words in the translated_sentence
        match the expected words from the map_dict.

        map_dict: Dictionary mapping words from the first language to the second.
        reverse_dict: Dictionary mapping words from the second language back to the first language.
        first_sentence: The sentence in the first language (string of space-separated words).
        translated_sentence: The translated sentence to be evaluated (string of space-separated words).
        
        Returns:
            The ratio of correctly translated words to total words in the sentence.
        """
        first_words = first_sentence.split()
        translated_words = translated_sentence.split()
        
        # Check if sentence lengths match
        if len(first_words) != len(translated_words) and interrupt_errors:
            raise ValueError("Both sentences must have the same number of words for accurate comparison.")
        
        # Count how many translations are correct
        correct_count = 0
        for first_word, translated_word in zip(first_words, translated_words):
            if map_dict.get(first_word) == translated_word:
                correct_count += 1
        
        # Return the ratio of correct translations
        return correct_count / len(first_words)


    def evaluate_model_accuracy(self, lang_map, reverse_map, sentence_len, inference, generate_limit, last_char_index, num_prompts=1, interrupt_errors = False):
        """
        Evaluates the model's translation accuracy by generating multiple prompts and testing them.

        Parameters:
            lang_map (dict): Mapping from the first language to the second language.
            reverse_map (dict): Reverse mapping from the second language to the first language.
            sentence_len (int): Maximum sentence length for generating language pairs.
            inference: Inference model or method to generate outputs.
            generate_limit (int): Maximum number of characters or tokens to generate.
            last_char_index (int): Index to slice the generated output to limit the number of characters or tokens.
            num_prompts (int, optional): Number of (X, Y) language translation pairs to use for generating the prompt. Default is 1.

        Returns:
            float: Translation accuracy of the model.
        """
        X_prompt = ""
        for _ in range(num_prompts):
            X, Y, _ = self.create_translate_language_instance(lang_map, reverse_map, sentence_len)
            X_prompt += X + " . " + Y + " | "

        x_test, y_test, _ = self.create_translate_language_instance(lang_map, reverse_map, sentence_len)
        X_test = X_prompt + x_test + " ."

        Y_output = inference.generate(X_test, generate_limit, k_value=1)[0 - last_char_index:]
        accuracy = self.test_translation_accuracy(lang_map, reverse_map, x_test, Y_output, interrupt_errors = interrupt_errors)
        
        return accuracy
