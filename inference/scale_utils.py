import numpy as np
from models.GPT import build_GPT

# Utils to work with estimation functions

def normalize_list(numbers):
    """
    Normalizes a list of numbers to the range [0, 1].
    
    Args:
        numbers (list of numeric): List of numbers to be normalized.
        
    Returns:
        list of float: Normalized list of numbers.

    Formula:
        $x_i = \frac{x_i - x_{min}}{x_{min} - x_{max}}$
        where, $x_{min}$ is the smallest value of all $x_i$
        $x_{max}$ is the largest value of all $x_i$ for all valid $i$.
    """
    min_val = min(numbers)
    max_val = max(numbers)
    normalized = [(x - min_val) / (max_val - min_val) for x in numbers]
    return normalized


def estimate_optimal_ratios_from_models(model_configs, 
                                       train_seq_len,
                                       x_train,
                                       y_train,
                                       max_epochs,
                                       batch_size):
    """
    Estimate the optimal ratios of model size and number of training tokens from FLOP counts.

    Args:
    - model_configs (list): List of tuples representing model configurations.
                            Each tuple contains parameters for building the model.
    - train_seq_len (list): List of integers representing different numbers of training sequences.
    - x_train (numpy array): Input data for training.
    - y_train (numpy array): Target data for training.
    - max_epochs (int): Maximum number of epochs for training.
    - batch_size (int): Batch size for training.

    Returns:
    - flops (numpy array): Array of FLOP counts for each experiment.
    - loss_history (numpy array): Array of loss histories for each experiment.
    - model_params (numpy array): Array of total model parameters for each experiment.

    Example:
        ```
        >>> # Define the Model configs to be tested for
        >>> vocab_size = 454+1
        >>> input_len = 256

        >>> model_configs =  [
        >>>     (input_len, vocab_size, 100, 1, 0, 10, 10, 1),
        >>>     (input_len, vocab_size, 200, 2, 0, 10, 20, 1),
        >>>     (input_len, vocab_size, 300, 1, 0, 30, 10, 1),
        >>>     (input_len, vocab_size, 400, 1, 0, 10, 40, 1),
        >>> ]

        >>> # Start testing the models by training them
        >>> model_epochs = estimate_optimal_ratios_from_models(model_configs, [1000, 2500, 5000], X[:5000], Y[:5000], 30, 128)

        >>> # Preprocessing numbers for plotting
        >>> flops = model_epochs[0]
        >>> loss_curve = model_epochs[1]
        >>> params = model_epochs[2]

        >>> flops_c = normalize_list(flops)

        >>> # Plotting
        >>> import matplotlib.pyplot as plt

        >>> fig = plt.figure(figsize=(10, 10), dpi=200)
        >>> for i in range(12):
        >>>     plt.plot(loss_curve[i], c=[flops_c[i], 0, 0], label=f'Floating-point Operations/forward inference: {flops[i]}' )
        >>> plt.legend()
        >>> plt.xlabel('Gradient Update Number #1')
        >>> plt.ylabel('Sparse Crossentropy Loss (with Logits)')
        ```
    """

    total_models = len(model_configs)
    total_seq_len = len(train_seq_len)

    print('Total Number of Experiments: ' + str(total_models * total_seq_len))

    experiment_number = 0
    _flops = []
    _loss_history = []
    _model_params = []
    for model_config in model_configs:
        for seq_len in train_seq_len:
            experiment_number += 1
            print('Train Number: ' + str(experiment_number))

            # Build the model and calculate FLOPs
            GPT, flops = build_GPT(*model_config)
            
            # Train the model
            history = GPT.fit(x_train[:seq_len], y_train[:seq_len], batch_size=batch_size, epochs=max_epochs)
            
            # Count model parameters
            model_params = GPT.count_params()

            # Extract loss history
            loss_history = history.history['loss']
            
            # Store results
            _flops.append(flops*seq_len*max_epochs)
            _loss_history.append(loss_history)
            _model_params.append(model_params)
    
    return (np.array(_flops), np.array(_loss_history), np.array(_model_params))

def estimate_optimal_ratios_from_flops(flop_list,
                                       input_len,
                                       num_heads,
                                       head_dims,
                                       num_decoders,
                                       fc_dim_factor,
                                       vocab_size,
                                       dropout_rate,
                                       x_train,
                                       y_train,
                                       trials_per_flop=2,
                                       batch_size=32):
    """
    Estimates optimal ratios of various model parameters based on FLOP count.

    Args:
    - flop_list (list): List of FLOP counts to estimate optimal ratios for.
    - input_len (int): Length of the input sequence.
    - num_heads (tuple): Tuple containing the minimum and maximum values for the number of attention heads.
    - head_dims (tuple): Tuple containing the minimum and maximum values for the dimensionality of attention heads.
    - num_decoders (int): Number of decoder layers.
    - fc_dim_factor (int): Factor to determine the dimensionality of fully connected layers.
    - vocab_size (int): Size of the vocabulary.
    - dropout_rate (float): Dropout rate.
    - x_train (numpy.ndarray): Training input data.
    - y_train (numpy.ndarray): Training target data.
    - trials_per_flop (int, optional): Number of trials per FLOP count. Defaults to 2.
    - batch_size (int, optional): Batch size for training. Defaults to 32.

    Warning:
    - **The `estimate_optimal_ratios_from_flops` is currently in the experimental phase
        and hasn't been tested thoroughly. It could lead to bugs!**

    Returns:
    - tuple: Tuple containing loss history, FLOP history, and number of parameters for each trial.
    """

    loss_history = []
    flop_history = []
    parameters = []

    for flop in flop_list:
        for _ in range(trials_per_flop):
            f_num_heads = np.random.randint(num_heads[0], num_heads[1])
            f_head_dims = np.random.randint(head_dims[0], head_dims[1])
            f_embed_dim = f_num_heads * f_head_dims
            f_num_decoders = np.random.randint(1, num_decoders)
            f_fc_dim_factor = np.random.randint(1, fc_dim_factor)

            args = (input_len,
                    vocab_size,
                    f_embed_dim,
                    f_num_decoders,
                    dropout_rate,
                    f_num_heads,
                    f_head_dims,
                    f_fc_dim_factor
                    )

            GPT, flop_per_inference = build_GPT(*args)  # Assuming build_GPT is defined elsewhere
            print(GPT.summary())

            epochs = flop // flop_per_inference
            if epochs <= 0:
                raise Exception('The provided FLOP count is too small: ' + str(flop) + ' is too small')

            history = GPT.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

            loss_history.append(history.history['loss'])
            flop_history.append(flop*batch_size*epochs)
            parameters.append(GPT.count_params())

    return loss_history, flop_history, parameters

