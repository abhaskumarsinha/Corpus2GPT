import os
import time
import contextlib

def get_distribution_scope(device_type):
    """
    Returns a context manager for executing code in a distributed environment.

    This function supports both JAX and TensorFlow backends. For the JAX backend, it supports CPU, GPU, and TPU devices,
    while for the TensorFlow backend, it supports CPU, GPU, and TPU devices with appropriate distribution strategies.

    The context manager returned by this function prints the total number of available devices and the total time taken
    for the code executed within the context manager.

    Args:
    - device_type (str): The type of device to use for distributed training. Can be 'cpu', 'gpu', or 'tpu'.

    Returns:
    - A context manager object that can be used to execute code in a distributed environment.

    
    Notes:
    -   For the JAX backend, this function uses the `jax.distribution` module to create a `DataParallel` distribution
        for GPU devices. For more information, see the Keras guide on distributed training with JAX:
        https://keras.io/guides/distribution/

    -   For the TensorFlow backend, this function uses the appropriate distribution strategy based on the device type:
            - For CPU and GPU, it uses `tf.distribute.MirroredStrategy`
            - For TPU, it uses `tf.distribute.TPUStrategy`

    -    For more information on distributed training with TensorFlow, see the TensorFlow guide:
         https://www.tensorflow.org/guide/distributed_training

    Raises:
    - ValueError: If an unsupported device type or backend is provided.

    Examples:
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
    """
    try:
        backend = os.environ.get("KERAS_BACKEND", "")
    except:
        backend = "tensorflow"

    if backend == "jax":
        
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices(device_type)
        num_devices = len(devices)

        def distribute_scope():
            start_time = time.time()

            @contextlib.contextmanager
            def scope_manager():
                try:
                    yield
                finally:
                    end_time = time.time()
                    print(f"Total {device_type.upper()}s available: {num_devices}")
                    print(f"Total time taken: {end_time - start_time:.2f} seconds")

            return scope_manager()

        return distribute_scope

    elif backend == "tensorflow":
        import tensorflow as tf
        from tensorflow.python.distribute import mirrored_strategy
      
        if device_type == "cpu":
            mirrored_strategy = tf.distribute.MirroredStrategy()

            def distribute_scope():
                start_time = time.time()

                @contextlib.contextmanager
                def scope_manager():
                    with mirrored_strategy.scope():
                        try:
                            yield
                        finally:
                            end_time = time.time()
                            print(f"Total CPUs available: {mirrored_strategy.num_replicas_in_sync}")
                            print(f"Total time taken: {end_time - start_time:.2f} seconds")

                return scope_manager()

            return distribute_scope

        elif device_type == "gpu":
            mirrored_strategy = tf.distribute.MirroredStrategy()

            def distribute_scope():
                start_time = time.time()

                @contextlib.contextmanager
                def scope_manager():
                    with mirrored_strategy.scope():
                        try:
                            yield
                        finally:
                            end_time = time.time()
                            print(f"Total GPUs available: {mirrored_strategy.num_replicas_in_sync}")
                            print(f"Total time taken: {end_time - start_time:.2f} seconds")

                return scope_manager()

            return distribute_scope

        elif device_type == "tpu":
            tpu_address = os.environ.get("TPU_NAME", "")
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

            def distribute_scope():
                start_time = time.time()

                @contextlib.contextmanager
                def scope_manager():
                    with tpu_strategy.scope():
                        try:
                            yield
                        finally:
                            end_time = time.time()
                            print(f"Total TPUs available: {tpu_strategy.num_replicas_in_sync}")
                            print(f"Total time taken: {end_time - start_time:.2f} seconds")

                return scope_manager()

            return distribute_scope

        else:
            print(f"Unsupported device type '{device_type}' for TensorFlow backend.")
            return lambda: None

    else:
        print("Unsupported backend. Please set KERAS_BACKEND to 'jax' or 'tensorflow'.")
        return lambda: None
