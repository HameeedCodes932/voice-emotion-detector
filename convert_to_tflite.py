import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("emotion_model.h5")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔧 Add these 2 lines to fix the TensorList error
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable default ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TF ops (like LSTM)
]
converter._experimental_lower_tensor_list_ops = False

# Convert
tflite_model = converter.convert()

# Save
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion to TFLite complete.")
