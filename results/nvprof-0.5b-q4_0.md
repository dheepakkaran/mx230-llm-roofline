total GPU kernel time 3128 ms over the whole run (128 tokens + prompt)
| kernel | ms per token | % of GPU time | calls per token |
|---|---|---|---|
| `mul_mat_vec_q<Q4_0, int=1, bool=1, bool=1, bool=0>` | 13.95 | 57.1% | 119 |
| `mul_mat_vec_q<Q8_0, int=1, bool=0, bool=1, bool=0>` | 4.68 | 19.1% | 1 |
| `mul_mat_vec_q<Q4_0, int=1, bool=1, bool=0, bool=0>` | 2.74 | 11.2% | 24 |
| `[CUDA memcpy HtoD]` | 0.97 | 4.0% | 8 |
| `flash_attn_tile<int=64, int=64, int=2, int=1, bool=0>` | 0.85 | 3.5% | 24 |
| `quantize_q8_1` | 0.30 | 1.2% | 146 |
| `[CUDA memcpy DtoH]` | 0.20 | 0.8% | 1 |
| `rms_norm_f32<float, int=256, int=1, bool=1, bool=1, bool=0, bool=0, fl` | 0.17 | 0.7% | 49 |
| `[CUDA memset]` | 0.12 | 0.5% | 1 |
| `mul_mat_vec_q<Q4_0, int=2, bool=0, bool=0, bool=0>` | 0.09 | 0.4% | 1 |
| `flash_attn_combine_results<int=64>` | 0.08 | 0.3% | 24 |
| `rope_neox<bool=1, bool=0, float, float>` | 0.08 | 0.3% | 24 |
