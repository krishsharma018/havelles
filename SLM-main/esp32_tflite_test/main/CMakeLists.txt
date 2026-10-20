idf_component_register(
    SRCS
        "esp32_tflite_test.cpp"
        "model/tiny_sine_int8_model.cc"
    INCLUDE_DIRS
        "."
        "model"
)