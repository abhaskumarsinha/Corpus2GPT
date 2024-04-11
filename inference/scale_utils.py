import numpy as np

def estimate_optimal_ratios_from_flops(model_configs, 
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
