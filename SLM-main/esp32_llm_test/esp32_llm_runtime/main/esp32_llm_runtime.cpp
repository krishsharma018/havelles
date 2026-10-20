#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"

#include "stories260K_model.h"
#include "tok512_model.h"
#include "esp_console.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
//#include "driver/uart.h"

static const char *TAG = "LLM_RUN";

// ============================================================
// STATIC INFERENCE WORKSPACE
// ============================================================

#define MAX_CONTEXT 64
#define TOKENIZER_VOCAB_SIZE 512
#define TOKENIZER_STORAGE_SIZE 7000
#define MAX_GENERATION_TOKENS 32
#define MAX_PROMPT_TOKENS 32
#define MAX_PROMPT_CHARS 128

static char g_prompt[MAX_PROMPT_CHARS];

static float g_x[64];
static float g_xb[64];
static float g_xb2[64];

static float g_hb[172];
static float g_hb2[172];

static float g_q[64];
static float g_k_current[32];
static float g_v_current[32];

static float g_att[MAX_CONTEXT];
static float g_logits[512];

static float g_key_cache[5 * MAX_CONTEXT * 32];
static float g_value_cache[5 * MAX_CONTEXT * 32];

// ============================================================
// STATIC TOKENIZER STORAGE
//
// The reference tokenizer uses malloc() for these structures.
// On ESP32 we keep them static instead.
// ============================================================

struct TinyTokenizer
{
    uint32_t max_token_length;

    float vocab_scores[TOKENIZER_VOCAB_SIZE];

    char *vocab[TOKENIZER_VOCAB_SIZE];

    // All vocabulary strings are copied here and terminated
    // with '\0'.
    char storage[TOKENIZER_STORAGE_SIZE];

    uint32_t storage_used;

    unsigned char byte_pieces[512];
};

static TinyTokenizer g_tokenizer;

// ============================================================
// Checkpoint helpers
// ============================================================

static uint32_t read_u32(const unsigned char *p)
{
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static int32_t read_i32(const unsigned char *p)
{
    int32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static float read_f32(const unsigned char *p)
{
    float value;
    memcpy(&value, p, sizeof(value));
    return value;
}

// ============================================================
// Tokenizer helpers
// ============================================================

static int tokenizer_lookup(
    const TinyTokenizer *t,
    const char *str)
{
    for (int i = 0; i < TOKENIZER_VOCAB_SIZE; i++)
    {
        if (strcmp(t->vocab[i], str) == 0)
        {
            return i;
        }
    }

    return -1;
}

// ============================================================
// Build tokenizer from embedded tok512.bin
//
// Exact tok512.bin format:
//
// int max_token_length
//
// for each vocab entry:
//     float score
//     int length
//     char[length] bytes
// ============================================================

static bool tokenizer_init(
    TinyTokenizer *t)
{
    memset(t, 0, sizeof(TinyTokenizer));

    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] =
            (unsigned char)i;

        t->byte_pieces[i * 2 + 1] =
            '\0';
    }

    const unsigned char *data =
        tok512_model;

    const uint32_t data_size =
        tok512_model_len;

    if (data_size < sizeof(uint32_t))
    {
        ESP_LOGE(
            TAG,
            "Tokenizer blob too small"
        );

        return false;
    }

    uint32_t offset = 0;

    // max_token_length
    t->max_token_length =
        read_u32(data + offset);

    offset += sizeof(uint32_t);

    ESP_LOGI(
        TAG,
        "tokenizer bytes = %u",
        data_size
    );

    ESP_LOGI(
        TAG,
        "max token length = %u",
        t->max_token_length
    );

    for (int i = 0;
         i < TOKENIZER_VOCAB_SIZE;
         i++)
    {
        // score
        if (offset + sizeof(float) >
            data_size)
        {
            ESP_LOGE(
                TAG,
                "Tokenizer truncated at score %d",
                i
            );

            return false;
        }

        t->vocab_scores[i] =
            read_f32(data + offset);

        offset += sizeof(float);

        // string length
        if (offset + sizeof(uint32_t) >
            data_size)
        {
            ESP_LOGE(
                TAG,
                "Tokenizer truncated at length %d",
                i
            );

            return false;
        }

        uint32_t len =
            read_u32(data + offset);

        offset += sizeof(uint32_t);

        if (offset + len >
            data_size)
        {
            ESP_LOGE(
                TAG,
                "Tokenizer string out of bounds %d",
                i
            );

            return false;
        }

        if (t->storage_used +
            len + 1 >
            TOKENIZER_STORAGE_SIZE)
        {
            ESP_LOGE(
                TAG,
                "Tokenizer storage overflow"
            );

            return false;
        }

        t->vocab[i] =
            &t->storage[t->storage_used];

        memcpy(
            t->vocab[i],
            data + offset,
            len
        );

        t->vocab[i][len] =
            '\0';

        t->storage_used +=
            len + 1;

        offset += len;
    }

    if (offset != data_size)
    {
        ESP_LOGE(
            TAG,
            "Tokenizer size mismatch: parsed=%u blob=%u",
            offset,
            data_size
        );

        return false;
    }

    ESP_LOGI(
        TAG,
        "tokenizer parse: PASS"
    );

    ESP_LOGI(
        TAG,
        "tokenizer storage used = %u bytes",
        t->storage_used
    );

    // Useful sanity checks.
    ESP_LOGI(
        TAG,
        "vocab[0] = \"%s\"",
        t->vocab[0]
    );

    ESP_LOGI(
        TAG,
        "vocab[1] = \"%s\"",
        t->vocab[1]
    );

    ESP_LOGI(
        TAG,
        "vocab[2] = \"%s\"",
        t->vocab[2]
    );

    return true;
}

// ============================================================
// Tokenizer encode
//
// Adapted directly from the reference run.c algorithm.
//
// This uses:
// - BOS = 1
// - EOS = 2
// - dummy prefix " "
// - UTF-8 codepoint handling
// - byte fallback
// - highest-score consecutive BPE merging
// ============================================================

static bool tokenizer_encode(
    TinyTokenizer *t,
    const char *text,
    bool bos,
    bool eos,
    int *tokens,
    int max_tokens,
    int *n_tokens)
{
    if (text == NULL)
    {
        ESP_LOGE(
            TAG,
            "cannot encode NULL text"
        );

        return false;
    }

    *n_tokens = 0;

    if (bos)
    {
        if (*n_tokens >= max_tokens)
            return false;

        tokens[(*n_tokens)++] = 1;
    }

    // Reference implementation adds a dummy
    // prefix space if the input isn't empty.
    if (text[0] != '\0')
    {
        int dummy_prefix =
            tokenizer_lookup(t, " ");

        if (dummy_prefix == -1)
        {
            ESP_LOGE(
                TAG,
                "Tokenizer missing dummy prefix"
            );

            return false;
        }

        if (*n_tokens >= max_tokens)
            return false;

        tokens[(*n_tokens)++] =
            dummy_prefix;
    }

    char str_buffer[128];

    size_t str_len = 0;

    const char *c = text;

    while (*c != '\0')
    {
        unsigned char current =
            (unsigned char)*c;

        // Start a new UTF-8 codepoint whenever
        // current byte isn't a continuation byte.
        if ((current & 0xC0) != 0x80)
        {
            str_len = 0;
        }

        if (str_len >=
            sizeof(str_buffer) - 1)
        {
            ESP_LOGE(
                TAG,
                "UTF-8 buffer overflow"
            );

            return false;
        }

        str_buffer[str_len++] =
            *c;

        str_buffer[str_len] =
            '\0';

        const unsigned char next =
            (unsigned char)c[1];

        if ((next & 0xC0) == 0x80 &&
            str_len < 4)
        {
            c++;
            continue;
        }

        int id =
            tokenizer_lookup(
                t,
                str_buffer
            );

        if (id != -1)
        {
            if (*n_tokens >= max_tokens)
            {
                ESP_LOGE(
                    TAG,
                    "Prompt exceeds MAX_CONTEXT"
                );

                return false;
            }

            tokens[(*n_tokens)++] =
                id;
        }
        else
        {
            // Byte fallback.
            for (size_t i = 0;
                 i < str_len;
                 i++)
            {
                if (*n_tokens >= max_tokens)
                {
                    ESP_LOGE(
                        TAG,
                        "Prompt exceeds MAX_CONTEXT"
                    );

                    return false;
                }

                tokens[(*n_tokens)++] =
                    (unsigned char)
                    str_buffer[i] + 3;
            }
        }

        str_len = 0;
        c++;
    }

    // ========================================================
    // BPE merge loop
    //
    // Exact logic from reference run.c.
    // ========================================================

    while (1)
    {
        float best_score =
            -1e10f;

        int best_id = -1;
        int best_idx = -1;

        for (int i = 0;
             i < (*n_tokens - 1);
             i++)
        {
            const char *a =
                t->vocab[tokens[i]];

            const char *b =
                t->vocab[tokens[i + 1]];

            size_t a_len =
                strlen(a);

            size_t b_len =
                strlen(b);

            if (a_len + b_len + 1 >=
                sizeof(str_buffer))
            {
                continue;
            }

            memcpy(
                str_buffer,
                a,
                a_len
            );

            memcpy(
                str_buffer + a_len,
                b,
                b_len
            );

            str_buffer[a_len + b_len] =
                '\0';

            int id =
                tokenizer_lookup(
                    t,
                    str_buffer
                );

            if (id != -1 &&
                t->vocab_scores[id] >
                best_score)
            {
                best_score =
                    t->vocab_scores[id];

                best_id =
                    id;

                best_idx =
                    i;
            }
        }

        if (best_idx == -1)
        {
            break;
        }

        tokens[best_idx] =
            best_id;

        for (int i = best_idx + 1;
             i < (*n_tokens - 1);
             i++)
        {
            tokens[i] =
                tokens[i + 1];
        }

        (*n_tokens)--;
    }

    if (eos)
    {
        if (*n_tokens >= max_tokens)
        {
            ESP_LOGE(
                TAG,
                "EOS would exceed token buffer"
            );

            return false;
        }

        tokens[(*n_tokens)++] = 2;
    }

    return true;
}

// ============================================================
// Tokenizer decode
//
// Exact semantics of reference decode():
// - strip leading whitespace after BOS
// - convert <0xXX> byte tokens
// ============================================================

static const char *tokenizer_decode(
    TinyTokenizer *t,
    int prev_token,
    int token)
{
    if (token < 0 ||
        token >= TOKENIZER_VOCAB_SIZE)
    {
        return "";
    }

    char *piece =
        t->vocab[token];

    // Following BOS, strip one leading space.
    if (prev_token == 1 &&
        piece[0] == ' ')
    {
        piece++;
    }

    // Byte fallback token representation:
    // <0x01>, <0x41>, etc.
    if (piece[0] == '<' &&
        piece[1] == '0' &&
        piece[2] == 'x' &&
        piece[3] != '\0' &&
        piece[4] != '\0' &&
        piece[5] == '>')
    {
        char hex[3];

        hex[0] = piece[3];
        hex[1] = piece[4];
        hex[2] = '\0';

        unsigned int value = 0;

        if (sscanf(
                hex,
                "%02X",
                &value) == 1)
        {
            return (const char *)
                &t->byte_pieces[value * 2];
        }
    }

    return piece;
}

// ============================================================
// RMSNorm
// ============================================================

static void rmsnorm(
    const float *x,
    const unsigned char *weight_bytes,
    float *out,
    int size)
{
    const float eps = 1e-5f;

    float sum_sq = 0.0f;

    for (int i = 0; i < size; i++)
    {
        sum_sq +=
            x[i] * x[i];
    }

    float mean_sq =
        sum_sq / (float)size;

    float rms =
        sqrtf(mean_sq + eps);

    float inv_rms =
        1.0f / rms;

    for (int i = 0; i < size; i++)
    {
        float weight =
            read_f32(
                weight_bytes +
                i * sizeof(float)
            );

        out[i] =
            x[i] *
            inv_rms *
            weight;
    }
}

// ============================================================
// Matrix multiplication
//
// out = W * x
//
// Weight layout:
// [out_dim][in_dim]
// ============================================================

static void matmul(
    const float *x,
    const unsigned char *weight_bytes,
    float *out,
    int out_dim,
    int in_dim)
{
    for (int i = 0;
     i < out_dim;
     i++)
    {
        float sum = 0.0f;

        for (int j = 0;
            j < in_dim;
            j++)
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

        if ((i & 63) == 63)
        {
            vTaskDelay(1);
        }
    }
}

// ============================================================
// Softmax
// ============================================================

static void softmax(
    float *x,
    int size)
{
    float max_val = x[0];

    for (int i = 1;
         i < size;
         i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0;
         i < size;
         i++)
    {
        x[i] =
            expf(x[i] - max_val);

        sum += x[i];
    }

    float inv_sum =
        1.0f / sum;

    for (int i = 0;
         i < size;
         i++)
    {
        x[i] *= inv_sum;
    }
}

// ============================================================
// Argmax
// ============================================================

static int argmax(
    const float *logits,
    int size)
{
    int best = 0;

    float best_value =
        logits[0];

    for (int i = 1;
         i < size;
         i++)
    {
        if (logits[i] >
            best_value)
        {
            best_value =
                logits[i];

            best = i;
        }
    }

    return best;
}

// ============================================================
// Complete transformer forward pass
// ============================================================

static int forward(
    const unsigned char *checkpoint,

    uint32_t embedding_offset,
    uint32_t rms_att_offset,
    uint32_t wq_offset,
    uint32_t wk_offset,
    uint32_t wv_offset,
    uint32_t wo_offset,
    uint32_t rms_ffn_offset,
    uint32_t w1_offset,
    uint32_t w2_offset,
    uint32_t w3_offset,
    uint32_t rms_final_offset,

    uint32_t rope_real_offset,
    uint32_t rope_imag_offset,

    uint32_t vocab_size,
    uint32_t dim,
    uint32_t hidden_dim,
    uint32_t n_layers,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t kv_dim,
    uint32_t kv_mul,
    uint32_t head_size,

    int token,
    int pos,

    float *x,
    float *xb,
    float *xb2,
    float *hb,
    float *hb2,
    float *q,
    float *k_cache,
    float *v_cache,
    float *att,
    float *logits)
{
    // ========================================================
    // Token embedding
    // ========================================================

    const unsigned char *embedding =
        checkpoint +
        embedding_offset +
        (uint32_t)token *
        dim *
        sizeof(float);

    for (uint32_t i = 0;
         i < dim;
         i++)
    {
        x[i] =
            read_f32(
                embedding +
                i * sizeof(float)
            );
    }

    // ========================================================
    // Transformer layers
    // ========================================================

    for (uint32_t l = 0;
         l < n_layers;
         l++)
    {
        // ----------------------------------------------------
        // Attention RMSNorm
        // ----------------------------------------------------

        rmsnorm(
            x,
            checkpoint +
            rms_att_offset +
            l * dim *
            sizeof(float),
            xb,
            (int)dim
        );

        // ----------------------------------------------------
        // Q
        // ----------------------------------------------------

        matmul(
            xb,
            checkpoint +
            wq_offset +
            l * dim * dim *
            sizeof(float),
            q,
            (int)dim,
            (int)dim
        );

        // ----------------------------------------------------
        // K
        // ----------------------------------------------------

        float *k_current =
            g_k_current;

        matmul(
            xb,
            checkpoint +
            wk_offset +
            l * dim * kv_dim *
            sizeof(float),
            k_current,
            (int)kv_dim,
            (int)dim
        );

        // ----------------------------------------------------
        // V
        // ----------------------------------------------------

        float *v_current =
            g_v_current;

        matmul(
            xb,
            checkpoint +
            wv_offset +
            l * dim * kv_dim *
            sizeof(float),
            v_current,
            (int)kv_dim,
            (int)dim
        );

        // ----------------------------------------------------
        // Store K/V
        // ----------------------------------------------------

        uint32_t cache_base =
            (l * MAX_CONTEXT + pos) *
            kv_dim;

        for (uint32_t i = 0;
             i < kv_dim;
             i++)
        {
            k_cache[
                cache_base + i
            ] =
                k_current[i];

            v_cache[
                cache_base + i
            ] =
                v_current[i];
        }

        // ====================================================
        // RoPE
        //
        // IMPORTANT:
        // Use the precomputed RoPE tables stored inside
        // stories260K.bin.
        //
        // This matches the reference checkpoint format.
        // ====================================================

        const unsigned char *rope_real =
            checkpoint +
            rope_real_offset +
            pos *
            (head_size / 2) *
            sizeof(float);

        const unsigned char *rope_imag =
            checkpoint +
            rope_imag_offset +
            pos *
            (head_size / 2) *
            sizeof(float);

        for (uint32_t i = 0;
             i < dim;
             i += 2)
        {
            uint32_t pair =
                (i % head_size) / 2;

            float fcr =
                read_f32(
                    rope_real +
                    pair *
                    sizeof(float)
                );

            float fci =
                read_f32(
                    rope_imag +
                    pair *
                    sizeof(float)
                );

            float q0 =
                q[i];

            float q1 =
                q[i + 1];

            q[i] =
                q0 * fcr -
                q1 * fci;

            q[i + 1] =
                q0 * fci +
                q1 * fcr;

            if (i < kv_dim)
            {
                float k0 =
                    k_cache[
                        cache_base + i
                    ];

                float k1 =
                    k_cache[
                        cache_base + i + 1
                    ];

                k_cache[
                    cache_base + i
                ] =
                    k0 * fcr -
                    k1 * fci;

                k_cache[
                    cache_base + i + 1
                ] =
                    k0 * fci +
                    k1 * fcr;
            }
        }

        // ====================================================
        // Multi-head attention
        // ====================================================

        for (uint32_t h = 0;
             h < n_heads;
             h++)
        {
            float *q_head =
                q +
                h * head_size;

            float *att_head =
                att;

            for (int t = 0;
                 t <= pos;
                 t++)
            {
                uint32_t key_base =
                    (l * MAX_CONTEXT + t) *
                    kv_dim;

                uint32_t kv_head =
                    h / kv_mul;

                const float *k_head =
                    k_cache +
                    key_base +
                    kv_head * head_size;

                float score = 0.0f;

                for (uint32_t i = 0;
                     i < head_size;
                     i++)
                {
                    score +=
                        q_head[i] *
                        k_head[i];
                }

                score /=
                    sqrtf(
                        (float)head_size
                    );

                att_head[t] =
                    score;
            }

            softmax(
                att_head,
                pos + 1
            );

            float *xb_head =
                xb2 +
                h * head_size;

            for (uint32_t i = 0;
                 i < head_size;
                 i++)
            {
                xb_head[i] =
                    0.0f;
            }

            for (int t = 0;
                 t <= pos;
                 t++)
            {
                uint32_t value_base =
                    (l * MAX_CONTEXT + t) *
                    kv_dim;

                uint32_t kv_head =
                    h / kv_mul;

                const float *v_head =
                    v_cache +
                    value_base +
                    kv_head * head_size;

                float a =
                    att_head[t];

                for (uint32_t i = 0;
                     i < head_size;
                     i++)
                {
                    xb_head[i] +=
                        a *
                        v_head[i];
                }
            }
        }

        // ====================================================
        // Wo
        // ====================================================

        matmul(
            xb2,
            checkpoint +
            wo_offset +
            l * dim * dim *
            sizeof(float),
            xb,
            (int)dim,
            (int)dim
        );

        // Residual
        for (uint32_t i = 0;
             i < dim;
             i++)
        {
            x[i] +=
                xb[i];
        }

        // ====================================================
        // FFN RMSNorm
        // ====================================================

        rmsnorm(
            x,
            checkpoint +
            rms_ffn_offset +
            l * dim *
            sizeof(float),
            xb,
            (int)dim
        );

        // ====================================================
        // W1
        // ====================================================

        matmul(
            xb,
            checkpoint +
            w1_offset +
            l * dim * hidden_dim *
            sizeof(float),
            hb,
            (int)hidden_dim,
            (int)dim
        );

        // ====================================================
        // W3
        // ====================================================

        matmul(
            xb,
            checkpoint +
            w3_offset +
            l * dim * hidden_dim *
            sizeof(float),
            hb2,
            (int)hidden_dim,
            (int)dim
        );

        // ====================================================
        // SwiGLU
        // ====================================================

        for (uint32_t i = 0;
             i < hidden_dim;
             i++)
        {
            float value =
                hb[i];

            float sigmoid =
                1.0f /
                (1.0f + expf(-value));

            value *=
                sigmoid;

            value *=
                hb2[i];

            hb[i] =
                value;
        }

        // ====================================================
        // W2
        // ====================================================

        matmul(
            hb,
            checkpoint +
            w2_offset +
            l * hidden_dim * dim *
            sizeof(float),
            xb,
            (int)dim,
            (int)hidden_dim
        );

        // Residual
        for (uint32_t i = 0;
             i < dim;
             i++)
        {
            x[i] +=
                xb[i];
        }
    }

    // ========================================================
    // Final RMSNorm
    // ========================================================

    rmsnorm(
        x,
        checkpoint +
        rms_final_offset,
        x,
        (int)dim
    );

    // ========================================================
    // Shared classifier
    // ========================================================

    const unsigned char *classifier =
        checkpoint +
        embedding_offset;

    matmul(
        x,
        classifier,
        logits,
        (int)vocab_size,
        (int)dim
    );

    return argmax(
        logits,
        (int)vocab_size
    );
}

// ============================================================
// app_main
// ============================================================

extern "C" void app_main(void)
{
    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "========================================"
    );

    ESP_LOGI(
        TAG,
        "TinyStories 260K - ESP32 GENERATION"
    );

    ESP_LOGI(
        TAG,
        "========================================"
    );

    const unsigned char *checkpoint =
        stories260K_model;

    uint32_t checkpoint_size =
        stories260K_model_len;

    // ========================================================
    // Read model configuration
    // ========================================================

    uint32_t dim =
        read_u32(
            checkpoint + 0
        );

    uint32_t hidden_dim =
        read_u32(
            checkpoint + 4
        );

    uint32_t n_layers =
        read_u32(
            checkpoint + 8
        );

    uint32_t n_heads =
        read_u32(
            checkpoint + 12
        );

    uint32_t n_kv_heads =
        read_u32(
            checkpoint + 16
        );

    int32_t vocab_raw =
        read_i32(
            checkpoint + 20
        );

    uint32_t seq_len =
        read_u32(
            checkpoint + 24
        );

    uint32_t vocab_size =
        vocab_raw < 0
            ? (uint32_t)(-vocab_raw)
            : (uint32_t)vocab_raw;

    uint32_t head_size =
        dim / n_heads;

    uint32_t kv_dim =
        (dim * n_kv_heads) /
        n_heads;

    uint32_t kv_mul =
        n_heads /
        n_kv_heads;

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "MODEL"
    );

    ESP_LOGI(
        TAG,
        "----------------------------------------"
    );

    ESP_LOGI(
        TAG,
        "checkpoint = %u bytes",
        checkpoint_size
    );

    ESP_LOGI(
        TAG,
        "dim        = %u",
        dim
    );

    ESP_LOGI(
        TAG,
        "hidden_dim = %u",
        hidden_dim
    );

    ESP_LOGI(
        TAG,
        "layers     = %u",
        n_layers
    );

    ESP_LOGI(
        TAG,
        "heads      = %u",
        n_heads
    );

    ESP_LOGI(
        TAG,
        "kv_heads   = %u",
        n_kv_heads
    );

    ESP_LOGI(
        TAG,
        "vocab      = %u",
        vocab_size
    );

    ESP_LOGI(
        TAG,
        "context    = %d",
        MAX_CONTEXT
    );

    // ========================================================
    // Tensor offsets
    // ========================================================

    uint32_t offset =
        7 * sizeof(uint32_t);

    uint32_t embedding_offset =
        offset;

    offset +=
        vocab_size *
        dim *
        sizeof(float);

    uint32_t rms_att_offset =
        offset;

    offset +=
        n_layers *
        dim *
        sizeof(float);

    uint32_t wq_offset =
        offset;

    offset +=
        n_layers *
        dim *
        dim *
        sizeof(float);

    uint32_t wk_offset =
        offset;

    offset +=
        n_layers *
        dim *
        kv_dim *
        sizeof(float);

    uint32_t wv_offset =
        offset;

    offset +=
        n_layers *
        dim *
        kv_dim *
        sizeof(float);

    uint32_t wo_offset =
        offset;

    offset +=
        n_layers *
        dim *
        dim *
        sizeof(float);

    uint32_t rms_ffn_offset =
        offset;

    offset +=
        n_layers *
        dim *
        sizeof(float);

    uint32_t w1_offset =
        offset;

    offset +=
        n_layers *
        dim *
        hidden_dim *
        sizeof(float);

    uint32_t w2_offset =
        offset;

    offset +=
        n_layers *
        hidden_dim *
        dim *
        sizeof(float);

    uint32_t w3_offset =
        offset;

    offset +=
        n_layers *
        dim *
        hidden_dim *
        sizeof(float);

    uint32_t rms_final_offset =
        offset;

    offset +=
        dim *
        sizeof(float);

    // ========================================================
    // RoPE tables
    // ========================================================

    uint32_t rope_real_offset =
        offset;

    offset +=
        seq_len *
        head_size / 2 *
        sizeof(float);

    uint32_t rope_imag_offset =
        offset;

    offset +=
        seq_len *
        head_size / 2 *
        sizeof(float);

    // ========================================================
    // Final checkpoint verification
    // ========================================================

    if (offset != checkpoint_size)
    {
        ESP_LOGE(
            TAG,
            "CHECKPOINT LAYOUT MISMATCH: %u != %u",
            offset,
            checkpoint_size
        );

        return;
    }

    ESP_LOGI(
        TAG,
        "checkpoint layout: PASS"
    );

    // ========================================================
    // Tokenizer
    // ========================================================

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "TOKENIZER"
    );

    ESP_LOGI(
        TAG,
        "----------------------------------------"
    );

    if (!tokenizer_init(
            &g_tokenizer))
    {
        ESP_LOGE(
            TAG,
            "Tokenizer initialization FAILED"
        );

        return;
    }

    // ========================================================
    // Clear KV cache
    // ========================================================

    memset(
        g_key_cache,
        0,
        sizeof(g_key_cache)
    );

    memset(
        g_value_cache,
        0,
        sizeof(g_value_cache)
    );

    // ========================================================
    // Prompt
    //
    // Keep the initial prompt short because our current
    // deployment uses MAX_CONTEXT = 8.
    // ========================================================

    printf("\n========================================\n");
    printf(" ESP32 TinyStories 260K\n");
    printf(" Context window: %d tokens\n", MAX_CONTEXT);
    printf(" Enter prompt:\n");
    printf("> ");
    fflush(stdout);

    int stdin_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, stdin_flags | O_NONBLOCK);

    int prompt_len = 0;

    while (prompt_len < MAX_PROMPT_CHARS - 1)
    {
        char ch;

        int len = read(STDIN_FILENO, &ch, 1);

        if (len <= 0)
        {
            continue;
        }

        if (ch == '\r' || ch == '\n')
        {
            break;
        }

        if (ch == '\b' || ch == 127)
        {
            if (prompt_len > 0)
            {
                prompt_len--;
                printf("\b \b");
                fflush(stdout);
            }

            continue;
        }

        g_prompt[prompt_len++] = ch;

        putchar(ch);
        fflush(stdout);
    }

    g_prompt[prompt_len] = '\0';

    printf("\n");

    const char *prompt = g_prompt;

    int prompt_tokens[
        MAX_PROMPT_TOKENS
    ];

    int prompt_count = 0;

    if (!tokenizer_encode(
            &g_tokenizer,
            prompt,
            true,
            false,
            prompt_tokens,
            MAX_PROMPT_TOKENS,
            &prompt_count))
    {
        ESP_LOGE(
            TAG,
            "Prompt encoding FAILED"
        );

        return;
    }

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "GENERATION"
    );

    ESP_LOGI(
        TAG,
        "----------------------------------------"
    );

    ESP_LOGI(
        TAG,
        "prompt = \"%s\"",
        prompt
    );

    ESP_LOGI(
        TAG,
        "prompt token count = %d",
        prompt_count
    );

    for (int i = 0;
         i < prompt_count;
         i++)
    {
        ESP_LOGI(
            TAG,
            "prompt token[%d] = %d",
            i,
            prompt_tokens[i]
        );
    }

    // ========================================================
    // Process prompt
    // ========================================================

    int prev_token = 1;

    int token =
        prompt_tokens[0];

    int current_position = 0;

    // The first token is processed normally.
    for (int i = 0;
         i < prompt_count;
         i++)
    {
        token =
            prompt_tokens[i];

        ESP_LOGI(
            TAG,
            "forward prompt position %d, token %d",
            current_position,
            token
        );

        int next_token =
            forward(
                checkpoint,

                embedding_offset,
                rms_att_offset,
                wq_offset,
                wk_offset,
                wv_offset,
                wo_offset,
                rms_ffn_offset,
                w1_offset,
                w2_offset,
                w3_offset,
                rms_final_offset,

                rope_real_offset,
                rope_imag_offset,

                vocab_size,
                dim,
                hidden_dim,
                n_layers,
                n_heads,
                n_kv_heads,
                kv_dim,
                kv_mul,
                head_size,

                token,
                current_position,

                g_x,
                g_xb,
                g_xb2,
                g_hb,
                g_hb2,
                g_q,

                g_key_cache,
                g_value_cache,

                g_att,
                g_logits
            );

        prev_token =
            token;

        token =
            next_token;

        current_position++;
    }

    // ========================================================
    // Print generated continuation
    // ========================================================

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "GENERATED TEXT"
    );

    ESP_LOGI(
        TAG,
        "----------------------------------------"
    );

    // Print prompt first.
    printf(
        "\n%s",
        prompt
    );

    int64_t generation_start_us = esp_timer_get_time();
    int generated_token_count = 0;

    // token currently contains the first generated token.
    for (int generated = 0;
        generated < MAX_GENERATION_TOKENS;
        generated++)
    {
        if (current_position >=
            MAX_CONTEXT)
        {
            break;
        }

        const char *piece =
            tokenizer_decode(
                &g_tokenizer,
                prev_token,
                token
            );

        printf(
            "%s",
            piece
        );

        fflush(stdout);

        // Count this generated token.
        generated_token_count++;

        // Stop on EOS.
        if (token == 2)
        {
            break;
        }


        int next_token =
            forward(
                checkpoint,

                embedding_offset,
                rms_att_offset,
                wq_offset,
                wk_offset,
                wv_offset,
                wo_offset,
                rms_ffn_offset,
                w1_offset,
                w2_offset,
                w3_offset,
                rms_final_offset,

                rope_real_offset,
                rope_imag_offset,

                vocab_size,
                dim,
                hidden_dim,
                n_layers,
                n_heads,
                n_kv_heads,
                kv_dim,
                kv_mul,
                head_size,

                token,
                current_position,

                g_x,
                g_xb,
                g_xb2,
                g_hb,
                g_hb2,
                g_q,

                g_key_cache,
                g_value_cache,

                g_att,
                g_logits
            );

        prev_token =
            token;

        token =
            next_token;

        current_position++;
    }


    int64_t generation_end_us = esp_timer_get_time();

    double generation_time_sec =
        (generation_end_us - generation_start_us) / 1000000.0;

    double tokens_per_second =
        generated_token_count / generation_time_sec;

    ESP_LOGI(
        TAG,
        "INFERENCE TIME : %.3f seconds",
        generation_time_sec
    );

    ESP_LOGI(
        TAG,
        "GENERATED TOKENS: %d",
        generated_token_count
    );

    ESP_LOGI(
        TAG,
        "TOKENS/SECOND  : %.3f",
        tokens_per_second
    );

    printf(
        "\n"
    );

    // ========================================================
    // RAM
    // ========================================================

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "RAM"
    );

    ESP_LOGI(
        TAG,
        "----------------------------------------"
    );

    ESP_LOGI(
        TAG,
        "free heap          = %lu",
        (unsigned long)
        esp_get_free_heap_size()
    );

    ESP_LOGI(
        TAG,
        "minimum free heap  = %lu",
        (unsigned long)
        esp_get_minimum_free_heap_size()
    );

    ESP_LOGI(
        TAG,
        "largest free block = %lu",
        (unsigned long)
        heap_caps_get_largest_free_block(
            MALLOC_CAP_8BIT
        )
    );

    ESP_LOGI(
        TAG,
        ""
    );

    ESP_LOGI(
        TAG,
        "========================================"
    );

    ESP_LOGI(
        TAG,
        "🔥 GENERATION-v1 COMPLETE"
    );

    ESP_LOGI(
        TAG,
        "========================================"
    );
}









// #include <stdio.h>
// #include <stdint.h>
// #include <string.h>
// #include <math.h>

// #include "esp_log.h"
// #include "esp_system.h"
// #include "esp_heap_caps.h"

// #include "stories260K_model.h"

// static const char *TAG = "LLM_RUN";
// // ============================================================
// // STATIC INFERENCE WORKSPACE
// // Keep large activation buffers off the FreeRTOS task stack.
// // TinyStories 260K: dim=64, hidden_dim=172, vocab=512.
// // ============================================================

// #define MAX_CONTEXT 8

// static float g_x[64];
// static float g_xb[64];
// static float g_xb2[64];

// static float g_hb[172];
// static float g_hb2[172];

// static float g_q[64];
// static float g_k_current[32];
// static float g_v_current[32];

// static float g_att[MAX_CONTEXT];
// static float g_logits[512];

// // 5 layers × 8 positions × 32 KV dimensions × K/V
// static float g_key_cache[5 * MAX_CONTEXT * 32];
// static float g_value_cache[5 * MAX_CONTEXT * 32];

// // ============================================================
// // Runtime configuration
// // ============================================================
// //
// // IMPORTANT:
// // We intentionally use a small runtime context.
// //
// // Full 512-token float32 KV cache:
// // ~655 KB
// //
// // That does not fit in the available ESP32 RAM.
// //
// // This first deployment therefore uses 8 tokens of context.
// //
// // Model weights remain completely unchanged.
// // ============================================================


// // ============================================================
// // Checkpoint helpers
// // ============================================================

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

// // ============================================================
// // RMSNorm
// // ============================================================

// static void rmsnorm(
//     const float *x,
//     const unsigned char *weight_bytes,
//     float *out,
//     int size)
// {
//     const float eps = 1e-5f;

//     float sum_sq = 0.0f;

//     for (int i = 0; i < size; i++)
//     {
//         sum_sq += x[i] * x[i];
//     }

//     float mean_sq =
//         sum_sq / (float)size;

//     float rms =
//         sqrtf(mean_sq + eps);

//     float inv_rms =
//         1.0f / rms;

//     for (int i = 0; i < size; i++)
//     {
//         float weight =
//             read_f32(
//                 weight_bytes +
//                 i * sizeof(float)
//             );

//         out[i] =
//             x[i] *
//             inv_rms *
//             weight;
//     }
// }

// // ============================================================
// // Matrix multiplication
// //
// // out = W * x
// //
// // Weight layout:
// // [out_dim][in_dim]
// // ============================================================

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
//             float weight =
//                 read_f32(
//                     weight_bytes +
//                     (i * in_dim + j) *
//                     sizeof(float)
//                 );

//             sum += weight * x[j];
//         }

//         out[i] = sum;
//     }
// }

// // ============================================================
// // Softmax
// // ============================================================

// static void softmax(
//     float *x,
//     int size)
// {
//     float max_val = x[0];

//     for (int i = 1; i < size; i++)
//     {
//         if (x[i] > max_val)
//         {
//             max_val = x[i];
//         }
//     }

//     float sum = 0.0f;

//     for (int i = 0; i < size; i++)
//     {
//         x[i] = expf(x[i] - max_val);
//         sum += x[i];
//     }

//     float inv_sum = 1.0f / sum;

//     for (int i = 0; i < size; i++)
//     {
//         x[i] *= inv_sum;
//     }
// }

// // ============================================================
// // Argmax
// // ============================================================

// static int argmax(
//     const float *logits,
//     int size)
// {
//     int best = 0;
//     float best_value = logits[0];

//     for (int i = 1; i < size; i++)
//     {
//         if (logits[i] > best_value)
//         {
//             best_value = logits[i];
//             best = i;
//         }
//     }

//     return best;
// }



// // ============================================================
// // Complete transformer forward pass
// //
// // One token at position `pos`.
// //
// // KV cache:
// //     [layer][position][kv_dim]
// //
// // This follows the reference llama2.c forward structure.
// // ============================================================

// static int forward(
//     const unsigned char *checkpoint,
//     uint32_t embedding_offset,
//     uint32_t rms_att_offset,
//     uint32_t wq_offset,
//     uint32_t wk_offset,
//     uint32_t wv_offset,
//     uint32_t wo_offset,
//     uint32_t rms_ffn_offset,
//     uint32_t w1_offset,
//     uint32_t w2_offset,
//     uint32_t w3_offset,
//     uint32_t rms_final_offset,
//     uint32_t vocab_size,
//     uint32_t dim,
//     uint32_t hidden_dim,
//     uint32_t n_layers,
//     uint32_t n_heads,
//     uint32_t n_kv_heads,
//     uint32_t kv_dim,
//     uint32_t kv_mul,
//     uint32_t head_size,
//     int token,
//     int pos,
//     float *x,
//     float *xb,
//     float *xb2,
//     float *hb,
//     float *hb2,
//     float *q,
//     float *k_cache,
//     float *v_cache,
//     float *att,
//     float *logits)
// {
//     // ========================================================
//     // Token embedding
//     // ========================================================

//     const unsigned char *embedding =
//         checkpoint +
//         embedding_offset +
//         (uint32_t)token *
//         dim *
//         sizeof(float);

//     for (uint32_t i = 0; i < dim; i++)
//     {
//         x[i] =
//             read_f32(
//                 embedding +
//                 i * sizeof(float)
//             );
//     }

//     // ========================================================
//     // Transformer layers
//     // ========================================================

//     for (uint32_t l = 0; l < n_layers; l++)
//     {
//         // ----------------------------------------------------
//         // Attention RMSNorm
//         // ----------------------------------------------------

//         rmsnorm(
//             x,
//             checkpoint +
//             rms_att_offset +
//             l * dim * sizeof(float),
//             xb,
//             (int)dim
//         );

//         // ----------------------------------------------------
//         // Q
//         // ----------------------------------------------------

//         matmul(
//             xb,
//             checkpoint +
//             wq_offset +
//             l * dim * dim * sizeof(float),
//             q,
//             (int)dim,
//             (int)dim
//         );

//         // ----------------------------------------------------
//         // K
//         // ----------------------------------------------------

//         float *k_current = g_k_current;

//         matmul(
//             xb,
//             checkpoint +
//             wk_offset +
//             l * dim * kv_dim * sizeof(float),
//             k_current,
//             (int)kv_dim,
//             (int)dim
//         );

//         // ----------------------------------------------------
//         // V
//         // ----------------------------------------------------

//         float *v_current = g_v_current;

//         matmul(
//             xb,
//             checkpoint +
//             wv_offset +
//             l * dim * kv_dim * sizeof(float),
//             v_current,
//             (int)kv_dim,
//             (int)dim
//         );

//         // ----------------------------------------------------
//         // Store current K/V in cache
//         // ----------------------------------------------------

//         uint32_t cache_base =
//             (l * MAX_CONTEXT + pos) *
//             kv_dim;

//         for (uint32_t i = 0; i < kv_dim; i++)
//         {
//             k_cache[cache_base + i] =
//                 k_current[i];

//             v_cache[cache_base + i] =
//                 v_current[i];
//         }

//         // ----------------------------------------------------
//         // RoPE
//         //
//         // Same mathematical implementation as reference.
//         // ----------------------------------------------------

//         for (uint32_t i = 0; i < dim; i += 2)
//         {
//             uint32_t head_dim =
//                 i % head_size;

//             float freq =
//                 1.0f /
//                 powf(
//                     10000.0f,
//                     head_dim /
//                     (float)head_size
//                 );

//             float angle =
//                 pos * freq;

//             float fcr =
//                 cosf(angle);

//             float fci =
//                 sinf(angle);

//             // Q is full dimension.
//             float q0 = q[i];
//             float q1 = q[i + 1];

//             q[i] =
//                 q0 * fcr -
//                 q1 * fci;

//             q[i + 1] =
//                 q0 * fci +
//                 q1 * fcr;

//             // K exists only for kv_dim dimensions.
//             if (i < kv_dim)
//             {
//                 float k0 =
//                     k_cache[cache_base + i];

//                 float k1 =
//                     k_cache[cache_base + i + 1];

//                 k_cache[cache_base + i] =
//                     k0 * fcr -
//                     k1 * fci;

//                 k_cache[cache_base + i + 1] =
//                     k0 * fci +
//                     k1 * fcr;
//             }
//         }

//         // ====================================================
//         // Multi-head attention
//         // ====================================================

//         for (uint32_t h = 0; h < n_heads; h++)
//         {
//             float *q_head =
//                 q +
//                 h * head_size;

//             float *att_head =
//                 att;

//             // ------------------------------------------------
//             // Q · K for every previous/current position
//             // ------------------------------------------------

//             for (int t = 0; t <= pos; t++)
//             {
//                 uint32_t key_base =
//                     (l * MAX_CONTEXT + t) *
//                     kv_dim;

//                 uint32_t kv_head =
//                     h / kv_mul;

//                 const float *k_head =
//                     k_cache +
//                     key_base +
//                     kv_head * head_size;

//                 float score = 0.0f;

//                 for (uint32_t i = 0;
//                      i < head_size;
//                      i++)
//                 {
//                     score +=
//                         q_head[i] *
//                         k_head[i];
//                 }

//                 score /=
//                     sqrtf(
//                         (float)head_size
//                     );

//                 att_head[t] =
//                     score;
//             }

//             // ------------------------------------------------
//             // Softmax
//             // ------------------------------------------------

//             softmax(
//                 att_head,
//                 pos + 1
//             );

//             // ------------------------------------------------
//             // Weighted sum of V
//             // ------------------------------------------------

//             float *xb_head =
//                 xb2 +
//                 h * head_size;

//             for (uint32_t i = 0;
//                  i < head_size;
//                  i++)
//             {
//                 xb_head[i] = 0.0f;
//             }

//             for (int t = 0; t <= pos; t++)
//             {
//                 uint32_t value_base =
//                     (l * MAX_CONTEXT + t) *
//                     kv_dim;

//                 uint32_t kv_head =
//                     h / kv_mul;

//                 const float *v_head =
//                     v_cache +
//                     value_base +
//                     kv_head * head_size;

//                 float a =
//                     att_head[t];

//                 for (uint32_t i = 0;
//                      i < head_size;
//                      i++)
//                 {
//                     xb_head[i] +=
//                         a *
//                         v_head[i];
//                 }
//             }
//         }

//         // ====================================================
//         // Output projection Wo
//         // ====================================================

//         matmul(
//             xb2,
//             checkpoint +
//             wo_offset +
//             l * dim * dim * sizeof(float),
//             xb,
//             (int)dim,
//             (int)dim
//         );

//         // ----------------------------------------------------
//         // Residual
//         // ----------------------------------------------------

//         for (uint32_t i = 0; i < dim; i++)
//         {
//             x[i] += xb[i];
//         }

//         // ====================================================
//         // FFN RMSNorm
//         // ====================================================

//         rmsnorm(
//             x,
//             checkpoint +
//             rms_ffn_offset +
//             l * dim * sizeof(float),
//             xb,
//             (int)dim
//         );

//         // ====================================================
//         // W1
//         // ====================================================

//         matmul(
//             xb,
//             checkpoint +
//             w1_offset +
//             l * dim * hidden_dim * sizeof(float),
//             hb,
//             (int)hidden_dim,
//             (int)dim
//         );

//         // ====================================================
//         // W3
//         // ====================================================

//         matmul(
//             xb,
//             checkpoint +
//             w3_offset +
//             l * dim * hidden_dim * sizeof(float),
//             hb2,
//             (int)hidden_dim,
//             (int)dim
//         );

//         // ====================================================
//         // SwiGLU
//         //
//         // silu(W1(x)) * W3(x)
//         // ====================================================

//         for (uint32_t i = 0;
//              i < hidden_dim;
//              i++)
//         {
//             float value =
//                 hb[i];

//             float sigmoid =
//                 1.0f /
//                 (1.0f + expf(-value));

//             value *=
//                 sigmoid;

//             value *=
//                 hb2[i];

//             hb[i] =
//                 value;
//         }

//         // ====================================================
//         // W2
//         // ====================================================

//         matmul(
//             hb,
//             checkpoint +
//             w2_offset +
//             l * hidden_dim * dim * sizeof(float),
//             xb,
//             (int)dim,
//             (int)hidden_dim
//         );

//         // ----------------------------------------------------
//         // Residual
//         // ----------------------------------------------------

//         for (uint32_t i = 0; i < dim; i++)
//         {
//             x[i] += xb[i];
//         }
//     }

//     // ========================================================
//     // Final RMSNorm
//     // ========================================================

//     rmsnorm(
//         x,
//         checkpoint +
//         rms_final_offset,
//         x,
//         (int)dim
//     );

//     // ========================================================
//     // Classifier
//     //
//     // Model has shared classifier weights.
//     //
//     // wcls = token_embedding_table
//     // ========================================================

//     const unsigned char *classifier =
//         checkpoint +
//         embedding_offset;

//     matmul(
//         x,
//         classifier,
//         logits,
//         (int)vocab_size,
//         (int)dim
//     );

//     // ========================================================
//     // Argmax
//     // ========================================================

//     return argmax(
//         logits,
//         (int)vocab_size
//     );
// }

// // ============================================================
// // app_main
// // ============================================================

// extern "C" void app_main(void)
// {
//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "========================================");
//     ESP_LOGI(TAG, "TinyStories 260K - ESP32 END-TO-END");
//     ESP_LOGI(TAG, "========================================");

//     const unsigned char *checkpoint =
//         stories260K_model;

//     uint32_t checkpoint_size =
//         stories260K_model_len;

//     // ========================================================
//     // Configuration
//     // ========================================================

//     uint32_t dim =
//         read_u32(checkpoint + 0);

//     uint32_t hidden_dim =
//         read_u32(checkpoint + 4);

//     uint32_t n_layers =
//         read_u32(checkpoint + 8);

//     uint32_t n_heads =
//         read_u32(checkpoint + 12);

//     uint32_t n_kv_heads =
//         read_u32(checkpoint + 16);

//     int32_t vocab_raw =
//         (int32_t)read_u32(checkpoint + 20);

//     uint32_t seq_len =
//         read_u32(checkpoint + 24);

//     uint32_t vocab_size =
//         vocab_raw < 0
//             ? (uint32_t)(-vocab_raw)
//             : (uint32_t)vocab_raw;

//     uint32_t head_size =
//         dim / n_heads;

//     uint32_t kv_dim =
//         (dim * n_kv_heads) /
//         n_heads;

//     uint32_t kv_mul =
//         n_heads /
//         n_kv_heads;

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "MODEL");
//     ESP_LOGI(TAG, "----------------------------------------");

//     ESP_LOGI(TAG,
//              "checkpoint = %u bytes",
//              checkpoint_size);

//     ESP_LOGI(TAG,
//              "dim        = %u",
//              dim);

//     ESP_LOGI(TAG,
//              "hidden_dim = %u",
//              hidden_dim);

//     ESP_LOGI(TAG,
//              "layers     = %u",
//              n_layers);

//     ESP_LOGI(TAG,
//              "heads      = %u",
//              n_heads);

//     ESP_LOGI(TAG,
//              "kv_heads   = %u",
//              n_kv_heads);

//     ESP_LOGI(TAG,
//              "vocab      = %u",
//              vocab_size);

//     ESP_LOGI(TAG,
//              "context    = %d",
//              MAX_CONTEXT);

//     // ========================================================
//     // Exact tensor offsets
//     // ========================================================

//     uint32_t offset =
//         7 * sizeof(uint32_t);

//     uint32_t embedding_offset =
//         offset;

//     offset +=
//         vocab_size *
//         dim *
//         sizeof(float);

//     uint32_t rms_att_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         sizeof(float);

//     uint32_t wq_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         dim *
//         sizeof(float);

//     uint32_t wk_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         kv_dim *
//         sizeof(float);

//     uint32_t wv_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         kv_dim *
//         sizeof(float);

//     uint32_t wo_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         dim *
//         sizeof(float);

//     uint32_t rms_ffn_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         sizeof(float);

//     uint32_t w1_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         hidden_dim *
//         sizeof(float);

//     uint32_t w2_offset =
//         offset;

//     offset +=
//         n_layers *
//         hidden_dim *
//         dim *
//         sizeof(float);

//     uint32_t w3_offset =
//         offset;

//     offset +=
//         n_layers *
//         dim *
//         hidden_dim *
//         sizeof(float);

//     uint32_t rms_final_offset =
//         offset;

//     offset +=
//         dim *
//         sizeof(float);

//     // Skip RoPE tables in checkpoint.
//     offset +=
//         seq_len *
//         head_size / 2 *
//         sizeof(float);

//     offset +=
//         seq_len *
//         head_size / 2 *
//         sizeof(float);

//     // ========================================================
//     // Layout check
//     // ========================================================

//     if (offset != checkpoint_size)
//     {
//         ESP_LOGE(
//             TAG,
//             "CHECKPOINT LAYOUT MISMATCH: %u != %u",
//             offset,
//             checkpoint_size
//         );

//         return;
//     }

//     ESP_LOGI(
//         TAG,
//         "checkpoint layout: PASS"
//     );

//     // // ========================================================
//     // // Runtime buffers
//     // // ========================================================

//     // float x[64];
//     // float xb[64];
//     // float xb2[64];

//     // float hb[172];
//     // float hb2[172];

//     // float q[64];

//     // float logits[512];

//     // float att[MAX_CONTEXT];

//     // ========================================================
//     // KV cache
//     //
//     // layers × context × kv_dim × 2
//     // ========================================================

//     // static float k_cache[
//     //     5 *
//     //     MAX_CONTEXT *
//     //     32
//     // ];

//     // static float v_cache[
//     //     5 *
//     //     MAX_CONTEXT *
//     //     32
//     // ];

//     memset(
//     g_key_cache,
//     0,
//     sizeof(g_key_cache)
// );

// memset(
//     g_value_cache,
//     0,
//     sizeof(g_value_cache)
// );

//     // ========================================================
//     // Initial token
//     //
//     // Token 1 is used as the first inference token.
//     //
//     // This is deliberately a raw-token end-to-end test.
//     // Tokenizer integration comes after the inference core.
//     // ========================================================

//     int token =
//         1;

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "========================================");
//     ESP_LOGI(TAG, "END-TO-END INFERENCE");
//     ESP_LOGI(TAG, "========================================");

//     ESP_LOGI(
//         TAG,
//         "initial token = %d",
//         token
//     );

//     // ========================================================
//     // Generate several tokens
//     // ========================================================

//     for (int pos = 0;
//          pos < MAX_CONTEXT;
//          pos++)
//     {
//         ESP_LOGI(
//             TAG,
//             "----------------------------------------"
//         );

//         ESP_LOGI(
//             TAG,
//             "forward position %d, token %d",
//             pos,
//             token
//         );

//         int next_token =
//             forward(
//                 checkpoint,

//                 embedding_offset,
//                 rms_att_offset,
//                 wq_offset,
//                 wk_offset,
//                 wv_offset,
//                 wo_offset,
//                 rms_ffn_offset,
//                 w1_offset,
//                 w2_offset,
//                 w3_offset,
//                 rms_final_offset,

//                 vocab_size,
//                 dim,
//                 hidden_dim,
//                 n_layers,
//                 n_heads,
//                 n_kv_heads,
//                 kv_dim,
//                 kv_mul,
//                 head_size,

//                 token,
//                 pos,

//                 g_x,
//                 g_xb,
//                 g_xb2,
//                 g_hb,
//                 g_hb2,
//                 g_q,

//                 g_key_cache,
//                 g_value_cache,

//                 g_att,
//                 g_logits
//             );

//         ESP_LOGI(
//             TAG,
//             "NEXT TOKEN = %d",
//             next_token
//         );

//         token =
//             next_token;
//     }

//     // ========================================================
//     // RAM
//     // ========================================================

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "RAM");
//     ESP_LOGI(TAG, "----------------------------------------");

//     ESP_LOGI(
//         TAG,
//         "free heap          = %lu",
//         (unsigned long)
//         esp_get_free_heap_size()
//     );

//     ESP_LOGI(
//         TAG,
//         "minimum free heap  = %lu",
//         (unsigned long)
//         esp_get_minimum_free_heap_size()
//     );

//     ESP_LOGI(
//         TAG,
//         "largest free block = %lu",
//         (unsigned long)
//         heap_caps_get_largest_free_block(
//             MALLOC_CAP_8BIT
//         )
//     );

//     // ========================================================
//     // Complete
//     // ========================================================

//     ESP_LOGI(TAG, "");
//     ESP_LOGI(TAG, "========================================");
//     ESP_LOGI(TAG, "🔥 END-TO-END INFERENCE COMPLETE");
//     ESP_LOGI(TAG, "========================================");
// }