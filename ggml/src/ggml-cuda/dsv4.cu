#include "dsv4.cuh"
#include "convert.cuh"
#include "ggml.h"

#include <cfloat>
#include <cstring>
#include <cstdint>

#define CUDA_DSV4_BLOCK_SIZE 256
#define CUDA_DSV4_FP8_BLOCK_SIZE 64

struct dsv4_rope_corr_dims {
    float v[2];
};

static __device__ __forceinline__ float dsv4_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

template<bool forward>
static __device__ __forceinline__ void dsv4_rope_yarn(
        const float theta_extrap,
        const float freq_scale,
        const dsv4_rope_corr_dims corr_dims,
        const int i0,
        const float ext_factor,
        float mscale,
        float & cos_theta,
        float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = dsv4_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool neox, typename T>
static __global__ void k_dsv4_rope_tail(
        const T * __restrict__ src0,
        T * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t d1,
        const int64_t d2,
        const int64_t d3,
        const int n_dims,
        const int n_nope,
        const int32_t * __restrict__ pos,
        const float freq_scale,
        const float ext_factor,
        const float attn_factor,
        const dsv4_rope_corr_dims corr_dims,
        const float theta_scale,
        const float * __restrict__ freq_factors) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= ne00) {
        return;
    }

    const int row = blockIdx.x;
    const int64_t i3 = row / (ne01 * ne02);
    const int64_t i2 = (row - i3 * ne01 * ne02) / ne01;
    const int64_t i1 = row - i3 * ne01 * ne02 - i2 * ne01;

    const int64_t src_row = i1 * s01 + i2 * s02 + i3 * s03;
    const int64_t dst_row = i1 * d1 + i2 * d2 + i3 * d3;

    if (d < n_nope) {
        dst[dst_row + d] = src0[src_row + d];
        return;
    }

    const int t = d - n_nope;
    const int pair_idx = neox ? (t % (n_dims / 2)) : (t / 2);
    const int i0 = neox ? 2 * pair_idx : 2 * pair_idx;
    const float theta_base = pos[i2] * powf(theta_scale, float(pair_idx));
    const float freq_factor = freq_factors ? freq_factors[pair_idx] : 1.0f;

    float cos_theta;
    float sin_theta;
    dsv4_rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    if constexpr (neox) {
        const int half = n_dims / 2;
        const int64_t x0_idx = src_row + n_nope + pair_idx;
        const int64_t x1_idx = x0_idx + half;
        const float x0 = src0[x0_idx];
        const float x1 = src0[x1_idx];
        const float out = t < half
            ? x0 * cos_theta - x1 * sin_theta
            : x0 * sin_theta + x1 * cos_theta;
        dst[dst_row + d] = ggml_cuda_cast<T>(out);
    } else {
        const int pair = 2 * pair_idx;
        const int64_t x0_idx = src_row + n_nope + pair;
        const int64_t x1_idx = x0_idx + 1;
        const float x0 = src0[x0_idx];
        const float x1 = src0[x1_idx];
        const float out = (t & 1) == 0
            ? x0 * cos_theta - x1 * sin_theta
            : x0 * sin_theta + x1 * cos_theta;
        dst[dst_row + d] = ggml_cuda_cast<T>(out);
    }
}

static __global__ void k_dsv4_hc_split_sinkhorn(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float * __restrict__ dst,
        const int64_t mixes_nb1,
        const int64_t dst_nb1,
        const int n_hc,
        const int sinkhorn_iters,
        const float eps,
        const int64_t n_rows) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const float * mix = (const float *) ((const char *) mixes + row * mixes_nb1);
    float * out = (float *) ((char *) dst + row * dst_nb1);

    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -FLT_MAX;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }

        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

static __global__ void k_dsv4_hc_weighted_sum(
        const char * __restrict__ x,
        const char * __restrict__ weights,
        char * __restrict__ dst,
        const int64_t x_nb0,
        const int64_t x_nb1,
        const int64_t x_nb2,
        const int64_t w_nb0,
        const int64_t w_nb1,
        const int64_t d_nb0,
        const int64_t d_nb1,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_tokens;
    if (i >= n_elem) {
        return;
    }

    const int64_t d = i % n_embd;
    const int64_t t = i / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        const float xv = *(const float *) (x + d * x_nb0 + h * x_nb1 + t * x_nb2);
        const float wv = *(const float *) (weights + h * w_nb0 + t * w_nb1);
        acc += xv * wv;
    }

    *(float *) (dst + d * d_nb0 + t * d_nb1) = acc;
}

static __global__ void k_dsv4_hc_expand(
        const char * __restrict__ block_out,
        const char * __restrict__ residual,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char * __restrict__ dst,
        const int64_t bo_nb0,
        const int64_t bo_nb1,
        const int64_t res_nb0,
        const int64_t res_nb1,
        const int64_t res_nb2,
        const int64_t post_nb0,
        const int64_t post_nb1,
        const int64_t comb_nb0,
        const int64_t comb_nb1,
        const int64_t comb_nb2,
        const int64_t dst_nb0,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    if (i >= n_elem) {
        return;
    }

    const int64_t d = i % n_embd;
    const int64_t tmp = i / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t = tmp / n_hc;

    const float block_v = *(const float *) (block_out + d * bo_nb0 + t * bo_nb1);
    const float post_v = *(const float *) (post + dst_hc * post_nb0 + t * post_nb1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *(const float *) (comb + dst_hc * comb_nb0 + src_hc * comb_nb1 + t * comb_nb2);
        const float res_v = *(const float *) (residual + d * res_nb0 + src_hc * res_nb1 + t * res_nb2);
        acc += comb_v * res_v;
    }

    *(float *) (dst + d * dst_nb0 + dst_hc * dst_nb1 + t * dst_nb2) = acc;
}

static __device__ __forceinline__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;

    for (int i = 1; i < 127; ++i) {
        const int exp = (i >> 3) & 0x0f;
        const int mant = i & 0x07;
        const float val = exp == 0
            ? ldexpf(float(mant), -9)
            : ldexpf(1.0f + float(mant) / 8.0f, exp - 7);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    const int exp = (best >> 3) & 0x0f;
    const int mant = best & 0x07;
    const float val = exp == 0
        ? ldexpf(float(mant), -9)
        : ldexpf(1.0f + float(mant) / 8.0f, exp - 7);

    return sign * val;
}

static __global__ void k_dsv4_fp8_kv_quantize_prefix(
        const float * __restrict__ src0,
        float * __restrict__ dst,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t n_segments_per_row,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t d1,
        const int64_t d2,
        const int64_t d3) {
    const int64_t seg = blockIdx.x;
    const int64_t row = seg / n_segments_per_row;
    const int64_t seg_in_row = seg - row * n_segments_per_row;
    const int64_t off = seg_in_row * CUDA_DSV4_FP8_BLOCK_SIZE;
    const int tid = threadIdx.x;

    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / (ne1 * ne2);

    const int64_t src_row = i1 * s1 + i2 * s2 + i3 * s3;
    const int64_t dst_row = i1 * d1 + i2 * d2 + i3 * d3;

    const float v = src0[src_row + off + tid];

    __shared__ float shared_reduce[CUDA_DSV4_FP8_BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_scale;

    float amax = fabsf(v);
    amax = block_reduce<block_reduce_method::MAX, CUDA_DSV4_FP8_BLOCK_SIZE>(amax, shared_reduce);

    if (tid == 0) {
        amax = fmaxf(amax, 1.0e-4f);
        const int exp = int(ceilf(log2f(amax / 448.0f)));
        shared_scale = ldexpf(1.0f, exp);
    }
    __syncthreads();

    const float scaled = fminf(fmaxf(v / shared_scale, -448.0f), 448.0f);
    dst[dst_row + off + tid] = dsv4_e4m3fn_dequant(scaled) * shared_scale;
}

static __global__ void k_dsv4_fp8_kv_copy_tail(
        const float * __restrict__ src0,
        float * __restrict__ dst,
    const int64_t tail_total,
        const int64_t head_dim,
        const int64_t n_rot,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t d1,
        const int64_t d2,
        const int64_t d3) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tail_total) {
        return;
    }

    const int64_t row = i / n_rot;
    const int64_t t = i - row * n_rot;
    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / (ne1 * ne2);

    const int64_t src_row = i1 * s1 + i2 * s2 + i3 * s3;
    const int64_t dst_row = i1 * d1 + i2 * d2 + i3 * d3;
    const int64_t off = head_dim - n_rot + t;
    dst[dst_row + off] = src0[src_row + off];
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_hc = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps = ggml_get_op_params_f32(dst, 2);
    const int64_t n_rows = ggml_nrows(mixes);
    const int64_t num_blocks = (n_rows + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE;

    k_dsv4_hc_split_sinkhorn<<<num_blocks, CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) mixes->data,
            (const float *) scale->data,
            (const float *) base->data,
            (float *) dst->data,
            mixes->nb[1],
            dst->nb[1],
            n_hc,
            sinkhorn_iters,
            eps,
            n_rows);
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_hc = x->ne[1];
    const int64_t n_elem = n_embd * n_tokens;
    const int64_t num_blocks = (n_elem + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE;

    k_dsv4_hc_weighted_sum<<<num_blocks, CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char *) x->data,
            (const char *) weights->data,
            (char *) dst->data,
            x->nb[0],
            x->nb[1],
            x->nb[2],
            weights->nb[0],
            weights->nb[1],
            dst->nb[0],
            dst->nb[1],
            n_embd,
            n_hc,
            n_tokens);
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post = dst->src[2];
    const ggml_tensor * comb = dst->src[3];

    const int64_t n_embd = dst->ne[0];
    const int64_t n_hc = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    const int64_t num_blocks = (n_elem + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE;

    k_dsv4_hc_expand<<<num_blocks, CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const char *) block_out->data,
            (const char *) residual->data,
            (const char *) post->data,
            (const char *) comb->data,
            (char *) dst->data,
            block_out->nb[0],
            block_out->nb[1],
            residual->nb[0],
            residual->nb[1],
            residual->nb[2],
            post->nb[0],
            post->nb[1],
            comb->nb[0],
            comb->nb[1],
            comb->nb[2],
            dst->nb[0],
            dst->nb[1],
            dst->nb[2],
            n_embd,
            n_hc,
            n_tokens);
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_rot = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src0->ne[0];
    const int64_t n_nope = head_dim - n_rot;
    const int64_t n_segments_per_row = n_nope / CUDA_DSV4_FP8_BLOCK_SIZE;
    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (n_segments_per_row > 0) {
        k_dsv4_fp8_kv_quantize_prefix<<<n_rows * n_segments_per_row, CUDA_DSV4_FP8_BLOCK_SIZE, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (float *) dst->data,
                src0->ne[1],
                src0->ne[2],
                n_segments_per_row,
                src0->nb[1] / sizeof(float),
                src0->nb[2] / sizeof(float),
                src0->nb[3] / sizeof(float),
                dst->nb[1] / sizeof(float),
                dst->nb[2] / sizeof(float),
                dst->nb[3] / sizeof(float));
    }

    if (n_rot > 0) {
        const int64_t tail_total = n_rows * n_rot;
        const int64_t num_blocks = (tail_total + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE;
        k_dsv4_fp8_kv_copy_tail<<<num_blocks, CUDA_DSV4_BLOCK_SIZE, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (float *) dst->data,
            tail_total,
                head_dim,
                n_rot,
                src0->ne[1],
                src0->ne[2],
                src0->nb[1] / sizeof(float),
                src0->nb[2] / sizeof(float),
                src0->nb[3] / sizeof(float),
                dst->nb[1] / sizeof(float),
                dst->nb[2] / sizeof(float),
                dst->nb[3] / sizeof(float));
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == src0->type);

    const int n_dims = ((const int32_t *) dst->op_params)[0];
    const int mode = ((const int32_t *) dst->op_params)[1];
    const int n_ctx_orig = ((const int32_t *) dst->op_params)[2];
    const bool inverse = ((const int32_t *) dst->op_params)[3] != 0;
    const int n_nope = src0->ne[0] - n_dims;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    dsv4_rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const int64_t nr = ggml_nrows(src0);
    const dim3 block_dims(CUDA_DSV4_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(nr, (src0->ne[0] + CUDA_DSV4_BLOCK_SIZE - 1) / CUDA_DSV4_BLOCK_SIZE, 1);

    const float * freq_factors = src2 ? (const float *) src2->data : nullptr;

    if (mode == GGML_ROPE_TYPE_NORMAL) {
        if (src0->type == GGML_TYPE_F32) {
            if (inverse) {
                k_dsv4_rope_tail<false, false><<<block_nums, block_dims, 0, ctx.stream()>>>(
                        (const float *) src0->data,
                        (float *) dst->data,
                        src0->ne[0],
                        src0->ne[1],
                        src0->ne[2],
                        src0->nb[1] / sizeof(float),
                        src0->nb[2] / sizeof(float),
                        src0->nb[3] / sizeof(float),
                        dst->nb[1] / sizeof(float),
                        dst->nb[2] / sizeof(float),
                        dst->nb[3] / sizeof(float),
                        n_dims,
                        n_nope,
                        (const int32_t *) src1->data,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        corr_dims,
                        theta_scale,
                        freq_factors);
            } else {
                k_dsv4_rope_tail<true, false><<<block_nums, block_dims, 0, ctx.stream()>>>(
                        (const float *) src0->data,
                        (float *) dst->data,
                        src0->ne[0],
                        src0->ne[1],
                        src0->ne[2],
                        src0->nb[1] / sizeof(float),
                        src0->nb[2] / sizeof(float),
                        src0->nb[3] / sizeof(float),
                        dst->nb[1] / sizeof(float),
                        dst->nb[2] / sizeof(float),
                        dst->nb[3] / sizeof(float),
                        n_dims,
                        n_nope,
                        (const int32_t *) src1->data,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        corr_dims,
                        theta_scale,
                        freq_factors);
            }
        } else {
            if (inverse) {
                k_dsv4_rope_tail<false, false><<<block_nums, block_dims, 0, ctx.stream()>>>(
                        (const half *) src0->data,
                        (half *) dst->data,
                        src0->ne[0],
                        src0->ne[1],
                        src0->ne[2],
                        src0->nb[1] / sizeof(half),
                        src0->nb[2] / sizeof(half),
                        src0->nb[3] / sizeof(half),
                        dst->nb[1] / sizeof(half),
                        dst->nb[2] / sizeof(half),
                        dst->nb[3] / sizeof(half),
                        n_dims,
                        n_nope,
                        (const int32_t *) src1->data,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        corr_dims,
                        theta_scale,
                        freq_factors);
            } else {
                k_dsv4_rope_tail<true, false><<<block_nums, block_dims, 0, ctx.stream()>>>(
                        (const half *) src0->data,
                        (half *) dst->data,
                        src0->ne[0],
                        src0->ne[1],
                        src0->ne[2],
                        src0->nb[1] / sizeof(half),
                        src0->nb[2] / sizeof(half),
                        src0->nb[3] / sizeof(half),
                        dst->nb[1] / sizeof(half),
                        dst->nb[2] / sizeof(half),
                        dst->nb[3] / sizeof(half),
                        n_dims,
                        n_nope,
                        (const int32_t *) src1->data,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        corr_dims,
                        theta_scale,
                        freq_factors);
            }
        }
        return;
    }

    GGML_ASSERT(mode == GGML_ROPE_TYPE_NEOX);
    if (src0->type == GGML_TYPE_F32) {
        if (inverse) {
            k_dsv4_rope_tail<false, true><<<block_nums, block_dims, 0, ctx.stream()>>>(
                    (const float *) src0->data,
                    (float *) dst->data,
                    src0->ne[0],
                    src0->ne[1],
                    src0->ne[2],
                    src0->nb[1] / sizeof(float),
                    src0->nb[2] / sizeof(float),
                    src0->nb[3] / sizeof(float),
                    dst->nb[1] / sizeof(float),
                    dst->nb[2] / sizeof(float),
                    dst->nb[3] / sizeof(float),
                    n_dims,
                    n_nope,
                    (const int32_t *) src1->data,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    corr_dims,
                    theta_scale,
                    freq_factors);
        } else {
            k_dsv4_rope_tail<true, true><<<block_nums, block_dims, 0, ctx.stream()>>>(
                    (const float *) src0->data,
                    (float *) dst->data,
                    src0->ne[0],
                    src0->ne[1],
                    src0->ne[2],
                    src0->nb[1] / sizeof(float),
                    src0->nb[2] / sizeof(float),
                    src0->nb[3] / sizeof(float),
                    dst->nb[1] / sizeof(float),
                    dst->nb[2] / sizeof(float),
                    dst->nb[3] / sizeof(float),
                    n_dims,
                    n_nope,
                    (const int32_t *) src1->data,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    corr_dims,
                    theta_scale,
                    freq_factors);
        }
    } else {
        if (inverse) {
            k_dsv4_rope_tail<false, true><<<block_nums, block_dims, 0, ctx.stream()>>>(
                    (const half *) src0->data,
                    (half *) dst->data,
                    src0->ne[0],
                    src0->ne[1],
                    src0->ne[2],
                    src0->nb[1] / sizeof(half),
                    src0->nb[2] / sizeof(half),
                    src0->nb[3] / sizeof(half),
                    dst->nb[1] / sizeof(half),
                    dst->nb[2] / sizeof(half),
                    dst->nb[3] / sizeof(half),
                    n_dims,
                    n_nope,
                    (const int32_t *) src1->data,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    corr_dims,
                    theta_scale,
                    freq_factors);
        } else {
            k_dsv4_rope_tail<true, true><<<block_nums, block_dims, 0, ctx.stream()>>>(
                    (const half *) src0->data,
                    (half *) dst->data,
                    src0->ne[0],
                    src0->ne[1],
                    src0->ne[2],
                    src0->nb[1] / sizeof(half),
                    src0->nb[2] / sizeof(half),
                    src0->nb[3] / sizeof(half),
                    dst->nb[1] / sizeof(half),
                    dst->nb[2] / sizeof(half),
                    dst->nb[3] / sizeof(half),
                    n_dims,
                    n_nope,
                    (const int32_t *) src1->data,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    corr_dims,
                    theta_scale,
                    freq_factors);
        }
    }
}
