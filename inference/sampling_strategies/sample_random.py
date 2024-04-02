import keras
import numpy as np

def random_sampling_strategy(outputs, pos_num=0, k_value=3):

    if len(keras.ops.shape(outputs)) == 3:
        outputs = outputs[0][pos_num]
    else:
        outputs = outputs[pos_num]
    
    values, indices = keras.ops.top_k(outputs, k=k_value)
    values = keras.ops.softmax(values, -1)
    values = keras.ops.convert_to_numpy(values)
    indices = keras.ops.convert_to_numpy(indices)

    return np.random.choice(indices, p=values)
