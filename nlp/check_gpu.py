import tensorflow as tf

# Check for GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        gpu_status = tf.config.experimental.get_device_details(gpu)
        device_name = gpu_status['device_name']
        print(f"GPU {i}: {device_name}")
else:
    print("No GPUs detected.")
