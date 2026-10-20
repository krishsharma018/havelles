// #include <stdio.h>
// #include "esp_log.h"

// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// #include "tiny_sine_int8_model.h"

// static const char *TAG = "TFLITE_TEST";

// constexpr int kTensorArenaSize = 20 * 1024;
// static uint8_t tensor_arena[kTensorArenaSize];

// extern "C" void app_main(void)
// {
//     ESP_LOGI(TAG, "ESP32 TFLite Micro inference test starting...");

//     const tflite::Model *model =
//         tflite::GetModel(tiny_sine_int8_model);

//     if (model == nullptr) {
//         ESP_LOGE(TAG, "Failed to load model!");
//         return;
//     }

//     ESP_LOGI(TAG, "Model loaded successfully.");

//     if (model->version() != TFLITE_SCHEMA_VERSION) {
//         ESP_LOGE(TAG, "Model schema mismatch.");
//         return;
//     }

//     static tflite::MicroMutableOpResolver<4> resolver;

//     resolver.AddFullyConnected();
//     resolver.AddRelu();

//     static tflite::MicroInterpreter interpreter(
//         model,
//         resolver,
//         tensor_arena,
//         kTensorArenaSize);

//     if (interpreter.AllocateTensors() != kTfLiteOk) {
//         ESP_LOGE(TAG, "AllocateTensors() failed!");
//         return;
//     }

//     ESP_LOGI(TAG, "AllocateTensors() succeeded.");

//     TfLiteTensor *input = interpreter.input(0);
//     TfLiteTensor *output = interpreter.output(0);

//     ESP_LOGI(TAG,
//              "Input: type=%d scale=%f zero_point=%d",
//              input->type,
//              input->params.scale,
//              input->params.zero_point);

//     ESP_LOGI(TAG,
//              "Output: type=%d scale=%f zero_point=%d",
//              output->type,
//              output->params.scale,
//              output->params.zero_point);

//     // Test input: x = 0.0
//     float x = 0.0f;

//     // Quantize float input to INT8.
//     int32_t quantized =
//         (int32_t)(x / input->params.scale) + input->params.zero_point;

//     if (quantized > 127) quantized = 127;
//     if (quantized < -128) quantized = -128;

//     input->data.int8[0] = (int8_t)quantized;

//     ESP_LOGI(TAG,
//              "Input x=%f -> quantized=%d",
//              x,
//              (int)input->data.int8[0]);

//     // Run the neural network.
//     if (interpreter.Invoke() != kTfLiteOk) {
//         ESP_LOGE(TAG, "Invoke() failed!");
//         return;
//     }

//     // Dequantize the INT8 output.
//     float result =
//         (output->data.int8[0] - output->params.zero_point)
//         * output->params.scale;

//     ESP_LOGI(TAG,
//              "Inference successful: output=%f",
//              result);
// }

// #include <stdio.h>
// #include "esp_log.h"
// #include "esp_timer.h"

// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/schema/schema_generated.h"

// #include "tiny_sine_int8_model.h"

// static const char *TAG = "TFLITE_TEST";

// constexpr int kTensorArenaSize = 20 * 1024;
// static uint8_t tensor_arena[kTensorArenaSize];

// extern "C" void app_main(void)
// {
//     ESP_LOGI(TAG, "ESP32 TFLite Micro latency test starting...");

//     const tflite::Model *model =
//         tflite::GetModel(tiny_sine_int8_model);

//     if (model == nullptr) {
//         ESP_LOGE(TAG, "Failed to load model!");
//         return;
//     }

//     ESP_LOGI(TAG, "Model loaded successfully.");

//     if (model->version() != TFLITE_SCHEMA_VERSION) {
//         ESP_LOGE(TAG, "Model schema mismatch.");
//         return;
//     }

//     static tflite::MicroMutableOpResolver<4> resolver;

//     resolver.AddFullyConnected();
//     resolver.AddRelu();

//     static tflite::MicroInterpreter interpreter(
//         model,
//         resolver,
//         tensor_arena,
//         kTensorArenaSize);

//     if (interpreter.AllocateTensors() != kTfLiteOk) {
//         ESP_LOGE(TAG, "AllocateTensors() failed!");
//         return;
//     }

//     ESP_LOGI(TAG, "AllocateTensors() succeeded.");

//     TfLiteTensor *input = interpreter.input(0);
//     TfLiteTensor *output = interpreter.output(0);

//     ESP_LOGI(TAG,
//              "Input: type=%d scale=%f zero_point=%d",
//              input->type,
//              input->params.scale,
//              input->params.zero_point);

//     ESP_LOGI(TAG,
//              "Output: type=%d scale=%f zero_point=%d",
//              output->type,
//              output->params.scale,
//              output->params.zero_point);

//     // ---------------------------------------------------------
//     // Warm-up inference
//     // ---------------------------------------------------------

//     float x = 0.0f;

//     int32_t quantized =
//         (int32_t)(x / input->params.scale)
//         + input->params.zero_point;

//     if (quantized > 127) quantized = 127;
//     if (quantized < -128) quantized = -128;

//     input->data.int8[0] = (int8_t)quantized;

//     if (interpreter.Invoke() != kTfLiteOk) {
//         ESP_LOGE(TAG, "Warm-up Invoke() failed!");
//         return;
//     }

//     ESP_LOGI(TAG, "Warm-up inference successful.");

//     // ---------------------------------------------------------
//     // Latency measurement
//     // ---------------------------------------------------------

//     constexpr int kRuns = 100;

//     int64_t total_us = 0;
//     int64_t min_us = INT64_MAX;
//     int64_t max_us = 0;

//     for (int i = 0; i < kRuns; i++) {

//         int64_t start = esp_timer_get_time();

//         TfLiteStatus status = interpreter.Invoke();

//         int64_t end = esp_timer_get_time();

//         if (status != kTfLiteOk) {
//             ESP_LOGE(TAG, "Invoke() failed at run %d!", i);
//             return;
//         }

//         int64_t latency_us = end - start;

//         total_us += latency_us;

//         if (latency_us < min_us) {
//             min_us = latency_us;
//         }

//         if (latency_us > max_us) {
//             max_us = latency_us;
//         }
//     }

//     float average_us =
//         (float)total_us / (float)kRuns;

//     // ---------------------------------------------------------
//     // Final result
//     // ---------------------------------------------------------

//     float result =
//         (output->data.int8[0] - output->params.zero_point)
//         * output->params.scale;

//     ESP_LOGI(TAG, "----------------------------------------");
//     ESP_LOGI(TAG, "Inference benchmark complete");
//     ESP_LOGI(TAG, "Runs       : %d", kRuns);
//     ESP_LOGI(TAG, "Min latency: %lld us", min_us);
//     ESP_LOGI(TAG, "Max latency: %lld us", max_us);
//     ESP_LOGI(TAG, "Avg latency: %.2f us", average_us);
//     ESP_LOGI(TAG, "Avg latency: %.3f ms", average_us / 1000.0f);
//     ESP_LOGI(TAG, "Final output: %f", result);
//     ESP_LOGI(TAG, "----------------------------------------");
// }

#include <stdio.h>
#include <stdint.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tiny_sine_int8_model.h"

static const char *TAG = "TFLITE_TEST";

constexpr int kTensorArenaSize = 20 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

static void print_heap(const char *stage)
{
    ESP_LOGI(TAG,
             "%s | Free heap: %lu bytes | Min free heap: %lu bytes",
             stage,
             (unsigned long)esp_get_free_heap_size(),
             (unsigned long)esp_get_minimum_free_heap_size());
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "ESP32 TFLite Micro RAM test starting...");

    // ---------------------------------------------------------
    // Baseline RAM
    // ---------------------------------------------------------

    print_heap("Before TFLite");

    const tflite::Model *model =
        tflite::GetModel(tiny_sine_int8_model);

    if (model == nullptr) {
        ESP_LOGE(TAG, "Failed to load model!");
        return;
    }

    ESP_LOGI(TAG, "Model loaded successfully.");

    print_heap("After model load");

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch.");
        return;
    }

    static tflite::MicroMutableOpResolver<4> resolver;

    resolver.AddFullyConnected();
    resolver.AddRelu();

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize);

    // ---------------------------------------------------------
    // Tensor allocation
    // ---------------------------------------------------------

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed!");
        return;
    }

    ESP_LOGI(TAG, "AllocateTensors() succeeded.");

    print_heap("After AllocateTensors");

    // ---------------------------------------------------------
    // Tensor information
    // ---------------------------------------------------------

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    ESP_LOGI(TAG,
             "Input: type=%d scale=%f zero_point=%d",
             input->type,
             input->params.scale,
             input->params.zero_point);

    ESP_LOGI(TAG,
             "Output: type=%d scale=%f zero_point=%d",
             output->type,
             output->params.scale,
             output->params.zero_point);

    // ---------------------------------------------------------
    // One inference
    // ---------------------------------------------------------

    float x = 0.0f;

    int32_t quantized =
        (int32_t)(x / input->params.scale)
        + input->params.zero_point;

    if (quantized > 127) quantized = 127;
    if (quantized < -128) quantized = -128;

    input->data.int8[0] = (int8_t)quantized;

    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed!");
        return;
    }

    ESP_LOGI(TAG, "Inference successful.");

    print_heap("After inference");

    // ---------------------------------------------------------
    // Tensor arena information
    // ---------------------------------------------------------

    ESP_LOGI(TAG,
             "Tensor arena reserved: %d bytes (%.2f KB)",
             kTensorArenaSize,
             kTensorArenaSize / 1024.0f);

    ESP_LOGI(TAG,
             "Tensor arena address: %p",
             (void *)tensor_arena);

    ESP_LOGI(TAG, "----------------------------------------");
    ESP_LOGI(TAG, "RAM benchmark complete");
    ESP_LOGI(TAG, "----------------------------------------");
}