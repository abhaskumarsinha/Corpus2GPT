[[`Corpus2GPT`](https://abhaskumarsinha.github.io/Corpus2GPT/)] [[`Docs`](https://abhaskumarsinha.github.io/Corpus2GPT/docs/doc/index.html)] [[`Ragdoll`](https://huggingface.co/spaces/abhaskumarsinha/Ragdoll)] [[`Keras`](https://www.github.com/keras-team/keras/)]

![Apache 2.0](https://badgen.net/static/license/Apache%202.0/blue) ![Keras 3.1](https://badgen.net/static/Keras/3.1/orange) ![Build: Pass](https://badgen.net/static/build/pass/f2a)

# Corpus2GPT
Corpus2GPT: A project enabling users to train their own GPT models on diverse datasets, including local languages and various corpus types, using Keras and compatible with TensorFlow, PyTorch, or JAX backends for subsequent storage or sharing.

# Work in Progress ⚠️

## Aim

Corpus2GPT is a pioneering project designed to empower users in training their own GPT models using diverse datasets, including those in local languages and various corpus types. Compatible with Keras and seamlessly supporting TensorFlow, PyTorch, or JAX backends, it stands out as one of the first tools in the field to offer this trifecta of backend options, facilitating benchmarking and flexibility for users. Beyond its initial capabilities, Corpus2GPT aspires to evolve into a comprehensive hub of language model tools, incorporating features like RAG (Retrieval-Augmented Generation) and MoEs (Mixture of Experts) in the future. With a commitment to staying at the forefront of LLM (Large Language Model) advancements, Corpus2GPT aims to become the go-to suite for both beginners and seasoned practitioners, offering accessible presets and modules for building cutting-edge language models.

## Why use Corpus2GPT?

### NLP Researchers and Practitioners:

- **Usage**: Train custom GPT models for various NLP tasks.
- **Contribution**: Provide feedback, suggest improvements, or contribute code.

### Data Scientists and ML Engineers:

- **Usage**: Build and fine-tune GPT models for specific tasks.
- **Contribution**: Implement new features, optimize algorithms.

### Language Enthusiasts and Linguists:

- **Usage**: Train models on diverse linguistic datasets.
- **Contribution**: Provide linguistic expertise, curate datasets.

### Educators and Students:

- **Usage**: Learn NLP concepts using Corpus2GPT.
- **Contribution**: Create educational resources, report bugs.

### Industry Professionals and Developers:

- **Usage**: Enhance language capabilities in applications.
- **Contribution**: Integrate Corpus2GPT, contribute to documentation.

## Current Features:

- **Classical Multihead Attention**: Corpus2GPT currently supports classical multihead attention mechanism, a key component in transformer architectures, aiding in capturing dependencies across different positions in the input sequences.
- **Decoder**: The tool includes a decoder module, essential for generating output sequences in autoregressive language models like GPT.
Random Sampling Search Strategies: Corpus2GPT implements random sampling search strategies, enabling users to generate diverse outputs during model inference.
- **Multiple Language Support**: With built-in support for multiple languages, Corpus2GPT facilitates training language models on diverse linguistic datasets, fostering inclusivity and accessibility.
- **Sentence Piece Tokenizer (and Vectorizer)**: Leveraging Sentence Piece Tokenizer and Vectorizer, Corpus2GPT offers efficient tokenization and vectorization of input data, crucial for preprocessing textual data in various languages and domains.
GPT Builder: Corpus2GPT provides a streamlined interface for building GPT models, simplifying the process of configuring and training custom language models.

## Upcoming Features:

- **MoE Support**: Future iterations of Corpus2GPT will incorporate support for Mixture of Experts (MoE), enhancing model capacity and enabling more efficient handling of complex data distributions.
- **Distributed Model Loading**: Plans are underway to introduce distributed loading of models across multiple devices such as GPUs and TPUs, optimizing training performance and scalability.
- **RAG Models**: Corpus2GPT aims to integrate Retrieval-Augmented Generation (RAG) models, enabling models to access external knowledge for improved generation tasks.
- **Other Search Strategies**: In addition to random sampling, Corpus2GPT will introduce additional search strategies for inference, offering users a range of options for controlling model outputs.
- **Transformer Debugger**: To enhance model analysis and interpretability, Corpus2GPT will introduce a Transformer Debugger tool, aiding users in understanding model behavior and performance.
- **Fine-Tuning Interface**: Implement a user-friendly interface for fine-tuning pre-trained GPT models on specific tasks or domains, allowing users to adapt models to their specific needs with ease.
- **Hyperparameter Optimization**: Incorporate automated hyperparameter optimization algorithms, such as Bayesian optimization or evolutionary strategies, to streamline the process of finding optimal model configurations for specific tasks or datasets.
- **Tree of Thoughts (Problem-Solving)**: Develop a Breadth-First Search (BFS) or Depth-First Search (DFS) based method to solve problems by decomposing them as a tree structure, enabling users to apply search algorithms to navigate through the problem space efficiently.
- **Model Distillation**: Implement model distillation techniques to transfer knowledge from large, complex models to smaller, more efficient ones, enabling the creation of compact yet high-performing language models suitable for deployment on resource-constrained devices or environments.


## Examples

### Creating a dataset from a list of files

```python
tokenizer = SPM_Tokenizer(vocab_size=5000, corpus='./stories.txt')
tokenizer = SPM_Tokenizer(vocab_model_file='./tokenizer_.model')

dataset = tokenizer.load_dataset(['./stories.txt'])

for (X, Y) in dataset:
    X = np.array(X)
    Y = np.array(Y)
```

### Creating a GPT Model

```python
GPT_Block = GPT(Decoder,
                TokenAndPositionEmbedding,
                vocab_size=5001,
                )

GPT_Model = keras.Sequential()
GPT_Model.add(keras.Input(shape=(64,)))
GPT_Model.add(GPT_Block)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

GPT_Model.compile(optimizer='adam', loss=[loss_fn])

GPT_Model.summary()
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ gpt (GPT)                            │ (None, 64, 5001)            │     127,668,361 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 127,668,361 (487.02 MB)
 Trainable params: 127,668,361 (487.02 MB)
 Non-trainable params: 0 (0.00 B)
```

### Inferencing from GPT Model

```python
inference = Generative_inference(model = model,
                          tokenizer = tokenizer,
                          search_strategy=random_sampling_strategy)
inference.generate("Hello World")
⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  Hello WorldAr things sayingWhen ruby...
```

### Distributed Learning on TPU
    ```python
    # JAX backend
    distribute_scope = get_distribution_scope("gpu")
    with distribute_scope():
        # Your code here
        # e.g., build and train a model
 

    # TensorFlow backend
    distribute_scope = get_distribution_scope("tpu")
    with distribute_scope():
        # Your code here
        # e.g., build and train a model
    ```


## Contributions are Welcome!

Corpus2GPT is an open-source project, and we welcome contributions from the community to help improve and enhance the tool. Whether you're an experienced developer, a domain expert, or someone passionate about language modeling and NLP, there are many ways you can contribute.





