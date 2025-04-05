import tensorflow as tf
from tensorflow.keras import layers
import math
from model.encoder.focal_svtr import FocalSVTR

def test_focalsvtr():
    input_tensor = tf.keras.Input(shape=(32, 128, 3))
    model = FocalSVTR(
        img_size=[32, 128],
        depths=[6, 6, 6],
        embed_dim=96,
        sub_k=[[1, 1], [2, 1], [1, 1]],
        focal_levels=[3, 3, 3],
        last_stage=False
    )

    # Chạy một lần để khởi tạo toàn bộ biến
    dummy_tensor = tf.random.normal((2, 32, 128, 3))
    _ = model(dummy_tensor)

    # Sau đó tạo hàm graph
    concrete_func = tf.function(model.call).get_concrete_function(
        dummy_tensor, training=False
    )

    # Chạy hàm graph
    output = concrete_func(dummy_tensor)
    print(f"Test passed! output shape = {output.shape}")

# Chạy hàm test
test_focalsvtr()