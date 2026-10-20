#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#include "stories260K_model.h"

static const char *TAG = "LLM_TEST";

// =========================================================
// Utility: read uint32 from checkpoint
// =========================================================

static uint32_t read_u32(const unsigned char *p)
{
    uint32_t value;

    memcpy(
        &value,
        p,
        sizeof(value)
    );

    return value;
}


// =========================================================
// Utility: read float32 from checkpoint
// =========================================================

static float read_f32(const unsigned char *p)
{
    float value;

    memcpy(
        &value,
        p,
        sizeof(value)
    );

    return value;
}


// =========================================================
// Utility: print tensor information
// =========================================================

static void print_tensor(
    const char *name,
    uint32_t offset,
    uint64_t elements)
{
    uint64_t bytes =
        elements * sizeof(float);

    ESP_LOGI(
        TAG,
        "%-24s offset=%7lu  elements=%8llu  bytes=%8llu",
        name,
        (unsigned long)offset,
        (unsigned long long)elements,
        (unsigned long long)bytes
    );
}


// =========================================================
// RMSNorm
// =========================================================

static void rmsnorm(
    const float *x,
    const float *weight,
    float *out,
    int size)
{
    const float eps = 1e-5f;

    float sum_sq = 0.0f;

    // Calculate sum of squares
    for (int i = 0; i < size; i++)
    {
        sum_sq += x[i] * x[i];
    }

    // Mean square
    float mean_sq =
        sum_sq / (float)size;

    // RMS
    float rms =
        sqrtf(mean_sq + eps);

    // Inverse RMS
    float inv_rms =
        1.0f / rms;

    // Normalize and scale
    for (int i = 0; i < size; i++)
    {
        out[i] =
            x[i] *
            inv_rms *
            weight[i];
    }
}


// =========================================================
// Matrix multiplication
//
// out = W * x
//
// W shape:
//     out_dim × in_dim
//
// x shape:
//     in_dim
//
// out shape:
//     out_dim
// =========================================================

static void matmul(
    const float *x,
    const unsigned char *weight_bytes,
    float *out,
    int out_dim,
    int in_dim)
{
    for (int i = 0; i < out_dim; i++)
    {
        float sum = 0.0f;

        for (int j = 0; j < in_dim; j++)
        {
            float weight =
                read_f32(
                    weight_bytes +
                    (i * in_dim + j) *
                    sizeof(float)
                );

            sum +=
                weight * x[j];
        }

        out[i] = sum;
    }
}


// =========================================================
// Application entry point
// =========================================================

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "TinyStories 260K ESP32 Runtime");
    ESP_LOGI(TAG, "========================================");


    // =====================================================
    // Checkpoint
    // =====================================================

    const unsigned char *checkpoint =
        stories260K_model;

    const uint32_t checkpoint_size =
        stories260K_model_len;

    ESP_LOGI(
        TAG,
        "Checkpoint address : %p",
        (const void *)checkpoint
    );

    ESP_LOGI(
        TAG,
        "Checkpoint size    : %u bytes",
        checkpoint_size
    );


    // =====================================================
    // Read model configuration
    // =====================================================

    uint32_t dim =
        read_u32(checkpoint + 0);

    uint32_t hidden_dim =
        read_u32(checkpoint + 4);

    uint32_t n_layers =
        read_u32(checkpoint + 8);

    uint32_t n_heads =
        read_u32(checkpoint + 12);

    uint32_t n_kv_heads =
        read_u32(checkpoint + 16);

    int32_t vocab_raw =
        (int32_t)read_u32(checkpoint + 20);

    uint32_t seq_len =
        read_u32(checkpoint + 24);


    // -----------------------------------------------------
    // Vocabulary size
    //
    // Positive vocab_raw:
    //     classifier weights are shared
    //
    // Negative vocab_raw:
    //     classifier has separate weights
    // -----------------------------------------------------

    uint32_t vocab_size =
        (vocab_raw < 0)
            ? (uint32_t)(-vocab_raw)
            : (uint32_t)vocab_raw;

    bool shared_weights =
        vocab_raw > 0;


    // =====================================================
    // Derived transformer dimensions
    // =====================================================

    uint32_t head_size =
        dim / n_heads;

    uint32_t kv_dim =
        (dim * n_kv_heads) / n_heads;

    uint32_t kv_mul =
        n_heads / n_kv_heads;


    // =====================================================
    // Print configuration
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "CONFIGURATION");
    ESP_LOGI(TAG, "-----------------------------------------------");

    ESP_LOGI(
        TAG,
        "dim          = %lu",
        (unsigned long)dim
    );

    ESP_LOGI(
        TAG,
        "hidden_dim   = %lu",
        (unsigned long)hidden_dim
    );

    ESP_LOGI(
        TAG,
        "n_layers     = %lu",
        (unsigned long)n_layers
    );

    ESP_LOGI(
        TAG,
        "n_heads      = %lu",
        (unsigned long)n_heads
    );

    ESP_LOGI(
        TAG,
        "n_kv_heads   = %lu",
        (unsigned long)n_kv_heads
    );

    ESP_LOGI(
        TAG,
        "vocab_size   = %lu",
        (unsigned long)vocab_size
    );

    ESP_LOGI(
        TAG,
        "seq_len      = %lu",
        (unsigned long)seq_len
    );

    ESP_LOGI(
        TAG,
        "head_size    = %lu",
        (unsigned long)head_size
    );

    ESP_LOGI(
        TAG,
        "kv_dim       = %lu",
        (unsigned long)kv_dim
    );

    ESP_LOGI(
        TAG,
        "kv_mul       = %lu",
        (unsigned long)kv_mul
    );

    ESP_LOGI(
        TAG,
        "shared wcls  = %s",
        shared_weights
            ? "YES"
            : "NO"
    );


    // =====================================================
    // Parse checkpoint tensor layout
    // =====================================================

    uint32_t offset =
        sizeof(uint32_t) * 7;


    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "WEIGHT LAYOUT");
    ESP_LOGI(TAG, "-----------------------------------------------");


    // =====================================================
    // 1. Token embedding table
    // =====================================================

    uint32_t embedding_offset =
        offset;

    uint64_t token_embedding_elements =
        (uint64_t)vocab_size * dim;

    print_tensor(
        "token_embedding_table",
        embedding_offset,
        token_embedding_elements
    );

    offset +=
        token_embedding_elements *
        sizeof(float);


    // =====================================================
    // 2. Attention RMSNorm
    // =====================================================

    uint32_t rms_att_offset =
        offset;

    uint64_t rms_att_elements =
        (uint64_t)n_layers * dim;

    print_tensor(
        "rms_att_weight",
        rms_att_offset,
        rms_att_elements
    );

    offset +=
        rms_att_elements *
        sizeof(float);


    // =====================================================
    // 3. Query projection WQ
    //
    // Shape per layer:
    //
    //     64 × 64
    //
    // Total:
    //
    //     n_layers × dim × dim
    // =====================================================

    uint32_t wq_offset =
        offset;

    uint64_t wq_elements =
        (uint64_t)n_layers *
        dim *
        (n_heads * head_size);

    print_tensor(
        "wq",
        wq_offset,
        wq_elements
    );

    offset +=
        wq_elements *
        sizeof(float);


    // =====================================================
    // 4. Key projection WK
    //
    // Shape per layer:
    //
    //     32 × 64
    // =====================================================

    uint32_t wk_offset =
        offset;

    uint64_t wk_elements =
        (uint64_t)n_layers *
        dim *
        (n_kv_heads * head_size);

    print_tensor(
        "wk",
        wk_offset,
        wk_elements
    );

    offset +=
        wk_elements *
        sizeof(float);


    // =====================================================
    // 5. Value projection WV
    //
    // Shape per layer:
    //
    //     32 × 64
    // =====================================================

    uint32_t wv_offset =
        offset;

    uint64_t wv_elements =
        (uint64_t)n_layers *
        dim *
        (n_kv_heads * head_size);

    print_tensor(
        "wv",
        wv_offset,
        wv_elements
    );

    offset +=
        wv_elements *
        sizeof(float);


    // =====================================================
    // 6. Output projection WO
    // =====================================================

    uint32_t wo_offset =
        offset;

    uint64_t wo_elements =
        (uint64_t)n_layers *
        (n_heads * head_size) *
        dim;

    print_tensor(
        "wo",
        wo_offset,
        wo_elements
    );

    offset +=
        wo_elements *
        sizeof(float);


    // =====================================================
    // 7. FFN RMSNorm
    // =====================================================

    uint32_t rms_ffn_offset =
        offset;

    uint64_t rms_ffn_elements =
        (uint64_t)n_layers * dim;

    print_tensor(
        "rms_ffn_weight",
        rms_ffn_offset,
        rms_ffn_elements
    );

    offset +=
        rms_ffn_elements *
        sizeof(float);


    // =====================================================
    // 8. FFN W1
    // =====================================================

    uint32_t w1_offset =
        offset;

    uint64_t w1_elements =
        (uint64_t)n_layers *
        dim *
        hidden_dim;

    print_tensor(
        "w1",
        w1_offset,
        w1_elements
    );

    offset +=
        w1_elements *
        sizeof(float);


    // =====================================================
    // 9. FFN W2
    // =====================================================

    uint32_t w2_offset =
        offset;

    uint64_t w2_elements =
        (uint64_t)n_layers *
        hidden_dim *
        dim;

    print_tensor(
        "w2",
        w2_offset,
        w2_elements
    );

    offset +=
        w2_elements *
        sizeof(float);


    // =====================================================
    // 10. FFN W3
    // =====================================================

    uint32_t w3_offset =
        offset;

    uint64_t w3_elements =
        (uint64_t)n_layers *
        dim *
        hidden_dim;

    print_tensor(
        "w3",
        w3_offset,
        w3_elements
    );

    offset +=
        w3_elements *
        sizeof(float);


    // =====================================================
    // 11. Final RMSNorm
    // =====================================================

    uint32_t rms_final_offset =
        offset;

    uint64_t rms_final_elements =
        dim;

    print_tensor(
        "rms_final_weight",
        rms_final_offset,
        rms_final_elements
    );

    offset +=
        rms_final_elements *
        sizeof(float);


    // =====================================================
    // RoPE tables
    // =====================================================

    uint32_t rope_real_offset =
        offset;

    uint64_t rope_elements =
        (uint64_t)seq_len *
        head_size / 2;

    print_tensor(
        "freq_cis_real",
        rope_real_offset,
        rope_elements
    );

    offset +=
        rope_elements *
        sizeof(float);


    uint32_t rope_imag_offset =
        offset;

    print_tensor(
        "freq_cis_imag",
        rope_imag_offset,
        rope_elements
    );

    offset +=
        rope_elements *
        sizeof(float);


    // =====================================================
    // Classifier
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "CLASSIFIER");
    ESP_LOGI(TAG, "-----------------------------------------------");

    if (shared_weights)
    {
        ESP_LOGI(
            TAG,
            "wcls -> shared token_embedding_table"
        );
    }
    else
    {
        uint32_t wcls_offset =
            offset;

        uint64_t wcls_elements =
            (uint64_t)vocab_size * dim;

        print_tensor(
            "wcls",
            wcls_offset,
            wcls_elements
        );

        offset +=
            wcls_elements *
            sizeof(float);
    }


    // =====================================================
    // Verify complete checkpoint layout
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "LAYOUT VERIFICATION");
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(
        TAG,
        "Calculated end : %lu bytes",
        (unsigned long)offset
    );

    ESP_LOGI(
        TAG,
        "Checkpoint size: %u bytes",
        checkpoint_size
    );

    if (offset == checkpoint_size)
    {
        ESP_LOGI(
            TAG,
            "PASS: calculated layout exactly matches checkpoint"
        );
    }
    else
    {
        ESP_LOGE(
            TAG,
            "FAIL: layout mismatch!"
        );

        ESP_LOGE(
            TAG,
            "Difference: %ld bytes",
            (long)checkpoint_size -
            (long)offset
        );

        return;
    }


    // =====================================================
    // Weight-value verification
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "WEIGHT VALUE VERIFICATION");
    ESP_LOGI(TAG, "-----------------------------------------------");

    ESP_LOGI(
        TAG,
        "Embedding offset : %lu",
        (unsigned long)embedding_offset
    );

    ESP_LOGI(
        TAG,
        "RMS att offset   : %lu",
        (unsigned long)rms_att_offset
    );

    ESP_LOGI(
        TAG,
        "WQ offset        : %lu",
        (unsigned long)wq_offset
    );

    ESP_LOGI(
        TAG,
        "WK offset        : %lu",
        (unsigned long)wk_offset
    );

    ESP_LOGI(
        TAG,
        "WV offset        : %lu",
        (unsigned long)wv_offset
    );


    // -----------------------------------------------------
    // Embedding values
    // -----------------------------------------------------

    ESP_LOGI(
        TAG,
        "Embedding[0] = %.9f",
        read_f32(
            checkpoint +
            embedding_offset +
            0 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "Embedding[1] = %.9f",
        read_f32(
            checkpoint +
            embedding_offset +
            1 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "Embedding[2] = %.9f",
        read_f32(
            checkpoint +
            embedding_offset +
            2 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "Embedding[3] = %.9f",
        read_f32(
            checkpoint +
            embedding_offset +
            3 * sizeof(float)
        )
    );


    // -----------------------------------------------------
    // WQ values
    // -----------------------------------------------------

    ESP_LOGI(
        TAG,
        "WQ[0] = %.9f",
        read_f32(
            checkpoint +
            wq_offset +
            0 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "WQ[1] = %.9f",
        read_f32(
            checkpoint +
            wq_offset +
            1 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "WQ[2] = %.9f",
        read_f32(
            checkpoint +
            wq_offset +
            2 * sizeof(float)
        )
    );

    ESP_LOGI(
        TAG,
        "WQ[3] = %.9f",
        read_f32(
            checkpoint +
            wq_offset +
            3 * sizeof(float)
        )
    );


    // =====================================================
    // RMSNorm verification
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "RMSNORM VERIFICATION");
    ESP_LOGI(TAG, "-----------------------------------------------");

    const int test_dim =
        (int)dim;

    float x[64];
    float rms_weight[64];
    float output[64];


    // -----------------------------------------------------
    // Load token embedding for token 0
    // -----------------------------------------------------

    for (int i = 0; i < test_dim; i++)
    {
        x[i] =
            read_f32(
                checkpoint +
                embedding_offset +
                i * sizeof(float)
            );
    }


    // -----------------------------------------------------
    // Load layer-0 attention RMSNorm weights
    // -----------------------------------------------------

    for (int i = 0; i < test_dim; i++)
    {
        rms_weight[i] =
            read_f32(
                checkpoint +
                rms_att_offset +
                i * sizeof(float)
            );
    }


    // -----------------------------------------------------
    // Execute RMSNorm
    // -----------------------------------------------------

    rmsnorm(
        x,
        rms_weight,
        output,
        test_dim
    );


    // -----------------------------------------------------
    // Print first 8 outputs
    // -----------------------------------------------------

    for (int i = 0; i < 8; i++)
    {
        ESP_LOGI(
            TAG,
            "y[%d] = %.12f",
            i,
            output[i]
        );
    }


    // =====================================================
    // Q / K / V PROJECTION
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Q / K / V PROJECTION VERIFICATION");
    ESP_LOGI(TAG, "-----------------------------------------------");


    // -----------------------------------------------------
    // Allocate projection outputs
    //
    // Q = dim = 64
    // K = kv_dim = 32
    // V = kv_dim = 32
    // -----------------------------------------------------

    float q[64];
    float k[32];
    float v[32];


    // -----------------------------------------------------
    // Q = WQ × normalized hidden state
    // -----------------------------------------------------

    matmul(
        output,
        checkpoint + wq_offset,
        q,
        (int)dim,
        (int)dim
    );


    // -----------------------------------------------------
    // K = WK × normalized hidden state
    // -----------------------------------------------------

    matmul(
        output,
        checkpoint + wk_offset,
        k,
        (int)kv_dim,
        (int)dim
    );


    // -----------------------------------------------------
    // V = WV × normalized hidden state
    // -----------------------------------------------------

    matmul(
        output,
        checkpoint + wv_offset,
        v,
        (int)kv_dim,
        (int)dim
    );


    // =====================================================
    // Print Q
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Q vector:");

    for (int i = 0; i < 16; i++)
    {
        ESP_LOGI(
            TAG,
            "q[%d] = %.12f",
            i,
            q[i]
        );
    }


    // =====================================================
    // Print K
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "K vector:");

    for (int i = 0; i < 16; i++)
    {
        ESP_LOGI(
            TAG,
            "k[%d] = %.12f",
            i,
            k[i]
        );
    }


    // =====================================================
    // Print V
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "V vector:");

    for (int i = 0; i < 16; i++)
    {
        ESP_LOGI(
            TAG,
            "v[%d] = %.12f",
            i,
            v[i]
        );
    }


    // =====================================================
    // RAM snapshot
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "RAM");
    ESP_LOGI(TAG, "-----------------------------------------------");

    ESP_LOGI(
        TAG,
        "Free heap          : %lu bytes",
        (unsigned long)
        esp_get_free_heap_size()
    );

    ESP_LOGI(
        TAG,
        "Minimum free heap  : %lu bytes",
        (unsigned long)
        esp_get_minimum_free_heap_size()
    );

    ESP_LOGI(
        TAG,
        "Largest free block : %lu bytes",
        (unsigned long)
        heap_caps_get_largest_free_block(
            MALLOC_CAP_8BIT
        )
    );

    ESP_LOGI(
        TAG,
        "Internal free RAM  : %lu bytes",
        (unsigned long)
        heap_caps_get_free_size(
            MALLOC_CAP_INTERNAL
        )
    );


    // =====================================================
    // Done
    // =====================================================

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Runtime verification complete.");
    ESP_LOGI(TAG, "========================================");
}


// // #include <stdio.h>
// // #include <stdint.h>
// // #include <string.h>

// // #include "esp_log.h"
// // #include "esp_system.h"
// // #include "esp_heap_caps.h"

// // #include "stories260K_model.h"

// // static const char *TAG = "LLM_TEST";

// // static uint32_t read_u32(const unsigned char *p)
// // {
// //     uint32_t value;
// //     memcpy(&value, p, sizeof(value));
// //     return value;
// // }

// // extern "C" void app_main(void)
// // {
// //     ESP_LOGI(TAG, "ESP32 TinyStories checkpoint parser");

// //     // ---------------------------------------------------------
// //     // Basic checkpoint information
// //     // ---------------------------------------------------------

// //     ESP_LOGI(TAG, "Checkpoint address : %p",
// //              (const void *)stories260K_model);

// //     ESP_LOGI(TAG, "Checkpoint size    : %u bytes",
// //              stories260K_model_len);

// //     // ---------------------------------------------------------
// //     // Read TinyStories checkpoint configuration
// //     // ---------------------------------------------------------

// //     const unsigned char *checkpoint = stories260K_model;

// //     uint32_t dim         = read_u32(checkpoint + 0);
// //     uint32_t hidden_dim  = read_u32(checkpoint + 4);
// //     uint32_t n_layers    = read_u32(checkpoint + 8);
// //     uint32_t n_heads     = read_u32(checkpoint + 12);
// //     uint32_t n_kv_heads  = read_u32(checkpoint + 16);
// //     uint32_t vocab_size  = read_u32(checkpoint + 20);
// //     uint32_t seq_len     = read_u32(checkpoint + 24);

// //     ESP_LOGI(TAG, "TinyStories configuration:");
// //     ESP_LOGI(TAG, "  dim         = %lu", (unsigned long)dim);
// //     ESP_LOGI(TAG, "  hidden_dim  = %lu", (unsigned long)hidden_dim);
// //     ESP_LOGI(TAG, "  n_layers    = %lu", (unsigned long)n_layers);
// //     ESP_LOGI(TAG, "  n_heads     = %lu", (unsigned long)n_heads);
// //     ESP_LOGI(TAG, "  n_kv_heads  = %lu", (unsigned long)n_kv_heads);
// //     ESP_LOGI(TAG, "  vocab_size  = %lu", (unsigned long)vocab_size);
// //     ESP_LOGI(TAG, "  seq_len     = %lu", (unsigned long)seq_len);

// //     // ---------------------------------------------------------
// //     // Derived transformer dimensions
// //     // ---------------------------------------------------------

// //     uint32_t head_size = dim / n_heads;
// //     uint32_t kv_dim = (dim * n_kv_heads) / n_heads;

// //     ESP_LOGI(TAG, "Derived dimensions:");
// //     ESP_LOGI(TAG, "  head_size   = %lu", (unsigned long)head_size);
// //     ESP_LOGI(TAG, "  kv_dim      = %lu", (unsigned long)kv_dim);

// //     // ---------------------------------------------------------
// //     // RAM snapshot
// //     // ---------------------------------------------------------

// //     ESP_LOGI(TAG, "RAM:");
// //     ESP_LOGI(TAG, "  Free heap          : %lu bytes",
// //              (unsigned long)esp_get_free_heap_size());

// //     ESP_LOGI(TAG, "  Minimum free heap  : %lu bytes",
// //              (unsigned long)esp_get_minimum_free_heap_size());

// //     ESP_LOGI(TAG, "  Largest free block : %lu bytes",
// //              (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

// //     ESP_LOGI(TAG, "  Internal free RAM  : %lu bytes",
// //              (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

// //     ESP_LOGI(TAG, "Checkpoint parsing complete.");
// // }

// #include <stdio.h>
// #include <stdint.h>
// #include <string.h>
// #include <math.h>

// #include "esp_log.h"
// #include "esp_system.h"
// #include "esp_heap_caps.h"

// #include "stories260K_model.h"

// static const char *TAG = "LLM_TEST";

// static uint32_t read_u32(const unsigned char *p)
// {
//     uint32_t value;
//     memcpy(&value, p, sizeof(value));
//     return value;
// }
// static float read_f32(const unsigned char *p)
// {
//     float value;
//     memcpy(&value, p, sizeof(value));
//     return value;
// }

// static void print_tensor(
//     const char *name,
//     uint32_t offset,
//     uint64_t elements)
// {
//     uint64_t bytes = elements * sizeof(float);

//     ESP_LOGI(
//         TAG,
//         "%-24s offset=%7lu  elements=%8llu  bytes=%8llu",
//         name,
//         (unsigned long)offset,
//         (unsigned long long)elements,
//         (unsigned long long)bytes
//     );
// }

// static void rmsnorm(
//     const float *x,
//     const float *weight,
//     float *out,
//     int size)
// {
//     const float eps = 1e-5f;

//     float sum_sq = 0.0f;

//     for (int i = 0; i < size; i++)
//     {
//         sum_sq += x[i] * x[i];
//     }

//     float mean_sq = sum_sq / (float)size;

//     float rms = sqrtf(mean_sq + eps);

//     float inv_rms = 1.0f / rms;

//     for (int i = 0; i < size; i++)
//     {
//         out[i] = x[i] * inv_rms * weight[i];
//     }
// }

// static void matmul(
//     const float *x,
//     const unsigned char *weight_bytes,
//     float *out,
//     int out_dim,
//     int in_dim)
// {
//     for (int i = 0; i < out_dim; i++)
//     {
//         float sum = 0.0f;

//         for (int j = 0; j < in_dim; j++)
//         {
//             float weight = read_f32(
//                 weight_bytes +
//                 (i * in_dim + j) * sizeof(float)
//             );

//             sum += weight * x[j];
//         }

//         out[i] = sum;
//     }
// }

// extern "C" void app_main(void)
// {
//     ESP_LOGI(TAG, "========================================");
//     ESP_LOGI(TAG, "TinyStories 260K weight-layout parser");
//     ESP_LOGI(TAG, "========================================");

//     const unsigned char *checkpoint = stories260K_model;

//     ESP_LOGI(TAG, "Checkpoint address : %p",
//              (const void *)checkpoint);

//     ESP_LOGI(TAG, "Checkpoint size    : %u bytes",
//              stories260K_model_len);

//     // ---------------------------------------------------------
//     // Read configuration
//     // ---------------------------------------------------------

//     uint32_t dim         = read_u32(checkpoint + 0);
//     uint32_t hidden_dim  = read_u32(checkpoint + 4);
//     uint32_t n_layers    = read_u32(checkpoint + 8);
//     uint32_t n_heads     = read_u32(checkpoint + 12);
//     uint32_t n_kv_heads  = read_u32(checkpoint + 16);
//     int32_t  vocab_raw   = (int32_t)read_u32(checkpoint + 20);
//     uint32_t seq_len     = read_u32(checkpoint + 24);

//     uint32_t vocab_size =
//         (vocab_raw < 0) ? (uint32_t)(-vocab_raw)
//                         : (uint32_t)vocab_raw;

//     uint32_t head_size = dim / n_heads;

//     uint32_t kv_dim =
//         (dim * n_kv_heads) / n_heads;

//     uint32_t kv_mul =
//         n_heads / n_kv_heads;

//     bool shared_weights = vocab_raw > 0;

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "CONFIGURATION");
//     ESP_LOGI(TAG, "  dim          = %lu",
//              (unsigned long)dim);

//     ESP_LOGI(TAG, "  hidden_dim   = %lu",
//              (unsigned long)hidden_dim);

//     ESP_LOGI(TAG, "  n_layers     = %lu",
//              (unsigned long)n_layers);

//     ESP_LOGI(TAG, "  n_heads      = %lu",
//              (unsigned long)n_heads);

//     ESP_LOGI(TAG, "  n_kv_heads   = %lu",
//              (unsigned long)n_kv_heads);

//     ESP_LOGI(TAG, "  vocab_size   = %lu",
//              (unsigned long)vocab_size);

//     ESP_LOGI(TAG, "  seq_len      = %lu",
//              (unsigned long)seq_len);

//     ESP_LOGI(TAG, "  head_size    = %lu",
//              (unsigned long)head_size);

//     ESP_LOGI(TAG, "  kv_dim       = %lu",
//              (unsigned long)kv_dim);

//     ESP_LOGI(TAG, "  kv_mul       = %lu",
//              (unsigned long)kv_mul);

//     ESP_LOGI(TAG, "  shared wcls  = %s",
//              shared_weights ? "YES" : "NO");

//     // ---------------------------------------------------------
//     // Parse weight layout
//     // ---------------------------------------------------------

//     uint32_t offset = sizeof(uint32_t) * 7;

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "WEIGHT LAYOUT");
//     ESP_LOGI(TAG, "-----------------------------------------------");

//     // 1. Token embedding table
//     uint64_t token_embedding_elements =
//         (uint64_t)vocab_size * dim;

//     print_tensor(
//         "token_embedding_table",
//         offset,
//         token_embedding_elements
//     );

//     offset += token_embedding_elements * sizeof(float);


//     // 2. Attention RMSNorm
//     uint64_t rms_att_elements =
//         (uint64_t)n_layers * dim;

//     print_tensor(
//         "rms_att_weight",
//         offset,
//         rms_att_elements
//     );

//     offset += rms_att_elements * sizeof(float);


//     // 3. Query projection
//     uint64_t wq_elements =
//         (uint64_t)n_layers *
//         dim *
//         (n_heads * head_size);

//     print_tensor(
//         "wq",
//         offset,
//         wq_elements
//     );

//     offset += wq_elements * sizeof(float);


//     // 4. Key projection
//     uint64_t wk_elements =
//         (uint64_t)n_layers *
//         dim *
//         (n_kv_heads * head_size);

//     print_tensor(
//         "wk",
//         offset,
//         wk_elements
//     );

//     offset += wk_elements * sizeof(float);


//     // 5. Value projection
//     uint64_t wv_elements =
//         (uint64_t)n_layers *
//         dim *
//         (n_kv_heads * head_size);

//     print_tensor(
//         "wv",
//         offset,
//         wv_elements
//     );

//     offset += wv_elements * sizeof(float);


//     // 6. Output projection
//     uint64_t wo_elements =
//         (uint64_t)n_layers *
//         (n_heads * head_size) *
//         dim;

//     print_tensor(
//         "wo",
//         offset,
//         wo_elements
//     );

//     offset += wo_elements * sizeof(float);


//     // 7. FFN RMSNorm
//     uint64_t rms_ffn_elements =
//         (uint64_t)n_layers * dim;

//     print_tensor(
//         "rms_ffn_weight",
//         offset,
//         rms_ffn_elements
//     );

//     offset += rms_ffn_elements * sizeof(float);


//     // 8. FFN W1
//     uint64_t w1_elements =
//         (uint64_t)n_layers *
//         dim *
//         hidden_dim;

//     print_tensor(
//         "w1",
//         offset,
//         w1_elements
//     );

//     offset += w1_elements * sizeof(float);


//     // 9. FFN W2
//     uint64_t w2_elements =
//         (uint64_t)n_layers *
//         hidden_dim *
//         dim;

//     print_tensor(
//         "w2",
//         offset,
//         w2_elements
//     );

//     offset += w2_elements * sizeof(float);


//     // 10. FFN W3
//     uint64_t w3_elements =
//         (uint64_t)n_layers *
//         dim *
//         hidden_dim;

//     print_tensor(
//         "w3",
//         offset,
//         w3_elements
//     );

//     offset += w3_elements * sizeof(float);


//     // 11. Final RMSNorm
//     uint64_t rms_final_elements = dim;

//     print_tensor(
//         "rms_final_weight",
//         offset,
//         rms_final_elements
//     );

//     offset += rms_final_elements * sizeof(float);


//     // ---------------------------------------------------------
//     // RoPE tables
//     // ---------------------------------------------------------

//     uint64_t rope_elements =
//         (uint64_t)seq_len * head_size / 2;

//     print_tensor(
//         "freq_cis_real (skip)",
//         offset,
//         rope_elements
//     );

//     offset += rope_elements * sizeof(float);


//     print_tensor(
//         "freq_cis_imag (skip)",
//         offset,
//         rope_elements
//     );

//     offset += rope_elements * sizeof(float);


//     // ---------------------------------------------------------
//     // Classifier
//     // ---------------------------------------------------------

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "CLASSIFIER");

//     if (shared_weights)
//     {
//         ESP_LOGI(
//             TAG,
//             "wcls -> shared token_embedding_table"
//         );
//     }
//     else
//     {
//         uint64_t wcls_elements =
//             (uint64_t)vocab_size * dim;

//         print_tensor(
//             "wcls",
//             offset,
//             wcls_elements
//         );

//         offset += wcls_elements * sizeof(float);
//     }


//     // ---------------------------------------------------------
//     // Final verification
//     // ---------------------------------------------------------

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "========================================");
//     ESP_LOGI(TAG, "LAYOUT VERIFICATION");
//     ESP_LOGI(TAG, "========================================");

//     ESP_LOGI(TAG,
//              "Calculated end : %lu bytes",
//              (unsigned long)offset);

//     ESP_LOGI(TAG,
//              "Checkpoint size: %u bytes",
//              stories260K_model_len);

//     if (offset == stories260K_model_len)
//     {
//         ESP_LOGI(TAG,
//                  "PASS: calculated layout exactly matches checkpoint");
//     }
//     else
//     {
//         ESP_LOGE(TAG,
//                  "FAIL: layout mismatch!");

//         ESP_LOGE(TAG,
//                  "Difference: %ld bytes",
//                  (long)stories260K_model_len - (long)offset);
//     }

// //     // ---------------------------------------------------------
// // // Weight-value verification
// // // ---------------------------------------------------------

// // ESP_LOGI(TAG, "");
// // ESP_LOGI(TAG, "WEIGHT VALUE VERIFICATION");
// // ESP_LOGI(TAG, "-----------------------------------------------");

// // // Token embedding starts immediately after the 7-word config header.
// // uint32_t embedding_offset = sizeof(uint32_t) * 7;

// // // WQ starts after:
// // //   token_embedding_table
// // //   rms_att_weight
// // uint32_t wq_offset =
// //     embedding_offset
// //     + (vocab_size * dim * sizeof(float))
// //     + (n_layers * dim * sizeof(float));

// // ESP_LOGI(TAG,
// //          "Embedding offset : %lu",
// //          (unsigned long)embedding_offset);

// // ESP_LOGI(TAG,
// //          "Wq offset        : %lu",
// //          (unsigned long)wq_offset);

// // // Read actual float32 values directly from the checkpoint bytes.
// // ESP_LOGI(TAG,
// //          "Embedding[0] = %.9f",
// //          read_f32(checkpoint + embedding_offset + 0 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Embedding[1] = %.9f",
// //          read_f32(checkpoint + embedding_offset + 1 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Embedding[2] = %.9f",
// //          read_f32(checkpoint + embedding_offset + 2 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Embedding[3] = %.9f",
// //          read_f32(checkpoint + embedding_offset + 3 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Wq[0]        = %.9f",
// //          read_f32(checkpoint + wq_offset + 0 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Wq[1]        = %.9f",
// //          read_f32(checkpoint + wq_offset + 1 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Wq[2]        = %.9f",
// //          read_f32(checkpoint + wq_offset + 2 * sizeof(float)));

// // ESP_LOGI(TAG,
// //          "Wq[3]        = %.9f",
// //          read_f32(checkpoint + wq_offset + 3 * sizeof(float)));
// // ---------------------------------------------------------
// // Weight-value verification
// // ---------------------------------------------------------

// ESP_LOGI(TAG, "");
// ESP_LOGI(TAG, "WEIGHT VALUE VERIFICATION");
// ESP_LOGI(TAG, "-----------------------------------------------");

// // Token embedding starts immediately after the 7-word config header.
// uint32_t embedding_offset = sizeof(uint32_t) * 7;

// ESP_LOGI(TAG,
//          "Embedding offset : %lu",
//          (unsigned long)embedding_offset);

// ESP_LOGI(TAG,
//          "Wq offset        : %lu",
//          (unsigned long)wq_offset);

// ESP_LOGI(TAG,
//          "Wk offset        : %lu",
//          (unsigned long)wk_offset);

// ESP_LOGI(TAG,
//          "Wv offset        : %lu",
//          (unsigned long)wv_offset);

// // Read actual float32 values directly from the checkpoint bytes.

// ESP_LOGI(TAG,
//          "Embedding[0] = %.9f",
//          read_f32(checkpoint + embedding_offset + 0 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Embedding[1] = %.9f",
//          read_f32(checkpoint + embedding_offset + 1 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Embedding[2] = %.9f",
//          read_f32(checkpoint + embedding_offset + 2 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Embedding[3] = %.9f",
//          read_f32(checkpoint + embedding_offset + 3 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Wq[0] = %.9f",
//          read_f32(checkpoint + wq_offset + 0 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Wq[1] = %.9f",
//          read_f32(checkpoint + wq_offset + 1 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Wq[2] = %.9f",
//          read_f32(checkpoint + wq_offset + 2 * sizeof(float)));

// ESP_LOGI(TAG,
//          "Wq[3] = %.9f",
//          read_f32(checkpoint + wq_offset + 3 * sizeof(float)));
    
//         // ---------------------------------------------------------
//     // RMSNorm verification
//     // ---------------------------------------------------------

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "RMSNORM VERIFICATION");
//     ESP_LOGI(TAG, "-----------------------------------------------");

//     const int test_dim = 64;

//     // Input = first 64 values of token embedding.
//     float x[test_dim];

//     // Layer-0 attention RMSNorm weights.
//     float rms_weight[test_dim];

//     float output[test_dim];

//     for (int i = 0; i < test_dim; i++)
//     {
//         x[i] = read_f32(
//             checkpoint +
//             embedding_offset +
//             i * sizeof(float)
//         );

//         rms_weight[i] = read_f32(
//             checkpoint +
//             131100 +
//             i * sizeof(float)
//         );
//     }

//     rmsnorm(
//         x,
//         rms_weight,
//         output,
//         test_dim
//     );

//     for (int i = 0; i < 8; i++)
//     {
//         ESP_LOGI(
//             TAG,
//             "y[%d] = %.12f",
//             i,
//             output[i]
//         );
//     }


//     //         // ---------------------------------------------------------
//     // // WQ MatMul verification
//     // // ---------------------------------------------------------

//     // ESP_LOGI(TAG, "");
//     // ESP_LOGI(TAG, "WQ MATMUL VERIFICATION");
//     // ESP_LOGI(TAG, "-----------------------------------------------");

//     // const int matmul_dim = 64;

//     // float q[matmul_dim];

//     // matmul(
//     //     output,
//     //     checkpoint + wq_offset,
//     //     q,
//     //     matmul_dim,
//     //     matmul_dim
//     // );

//     // for (int i = 0; i < 16; i++)
//     // {
//     //     ESP_LOGI(
//     //         TAG,
//     //         "q[%d] = %.12f",
//     //         i,
//     //         q[i]
//     //     );
//     // }
//     // ---------------------------------------------------------
// // Q / K / V PROJECTION VERIFICATION
// // ---------------------------------------------------------

// ESP_LOGI(TAG, "");
// ESP_LOGI(TAG, "Q / K / V PROJECTION VERIFICATION");
// ESP_LOGI(TAG, "-----------------------------------------------");

// const int q_dim = dim;
// const int kv_dim_runtime = kv_dim;

// float q[64];
// float k[32];
// float v[32];

// // Q = WQ * normalized_hidden
// matmul(
//     output,
//     checkpoint + wq_offset,
//     q,
//     q_dim,
//     dim
// );

// // K = WK * normalized_hidden
// matmul(
//     output,
//     checkpoint + wk_offset,
//     k,
//     kv_dim_runtime,
//     dim
// );

// // V = WV * normalized_hidden
// matmul(
//     output,
//     checkpoint + wv_offset,
//     v,
//     kv_dim_runtime,
//     dim
// );

// ESP_LOGI(TAG, "");
// ESP_LOGI(TAG, "Q vector:");
// for (int i = 0; i < 16; i++)
// {
//     ESP_LOGI(TAG, "q[%d] = %.12f", i, q[i]);
// }

// ESP_LOGI(TAG, "");
// ESP_LOGI(TAG, "K vector:");
// for (int i = 0; i < 16; i++)
// {
//     ESP_LOGI(TAG, "k[%d] = %.12f", i, k[i]);
// }

// ESP_LOGI(TAG, "");
// ESP_LOGI(TAG, "V vector:");
// for (int i = 0; i < 16; i++)
// {
//     ESP_LOGI(TAG, "v[%d] = %.12f", i, v[i]);
// }
    
    
    
//          // ---------------------------------------------------------
//     // RAM snapshot
//     // ---------------------------------------------------------

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "RAM");

//     ESP_LOGI(TAG,
//              "Free heap          : %lu bytes",
//              (unsigned long)esp_get_free_heap_size());

//     ESP_LOGI(TAG,
//              "Minimum free heap  : %lu bytes",
//              (unsigned long)esp_get_minimum_free_heap_size());

//     ESP_LOGI(TAG,
//              "Largest free block : %lu bytes",
//              (unsigned long)heap_caps_get_largest_free_block(
//                  MALLOC_CAP_8BIT
//              ));

//     ESP_LOGI(TAG,
//              "Internal free RAM  : %lu bytes",
//              (unsigned long)heap_caps_get_free_size(
//                  MALLOC_CAP_INTERNAL
//              ));

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "Parser finished.");
// }

