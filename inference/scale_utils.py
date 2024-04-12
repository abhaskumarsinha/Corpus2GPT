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
            _flops.append(flops)
            _loss_history.append(loss_history)
            _model_params.append(model_params)
    
    return (np.array(_flops), np.array(_loss_history), np.array(_model_params))

import numpy as np

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
        flop_list (list): List of FLOP counts to estimate optimal ratios for.
        input_len (int): Length of the input sequence.
        num_heads (tuple): Tuple containing the minimum and maximum values for the number of attention heads.
        head_dims (tuple): Tuple containing the minimum and maximum values for the dimensionality of attention heads.
        num_decoders (int): Number of decoder layers.
        fc_dim_factor (int): Factor to determine the dimensionality of fully connected layers.
        vocab_size (int): Size of the vocabulary.
        dropout_rate (float): Dropout rate.
        x_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training target data.
        trials_per_flop (int, optional): Number of trials per FLOP count. Defaults to 2.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        tuple: Tuple containing loss history, FLOP history, and number of parameters for each trial.
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
            flop_history.append(flop)
            parameters.append(GPT.count_params())

    return loss_history, flop_history, parameters

