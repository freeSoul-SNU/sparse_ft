# Sparse Fine-Tuning Methods — Unified Benchmark

5가지 Sparse Fine-Tuning 방법을 LMFlow 기반으로 통합 구현하여 MT-Bench, MMLU, CSR 벤치마크에서 비교 평가합니다.

## Methods

| Method | Paper/Repo | Sparsity Type | Description |
|--------|-----------|---------------|-------------|
| **SIFT** | [song-wx/SIFT](https://github.com/song-wx/SIFT) | Gradient top-k element-wise | calibration gradient의 absolute value 상위 weight element를 선택하여 학습 |
| **SpiEL** | [ducdauge/sft-llm](https://github.com/ducdauge/sft-llm) + [AlanAnsell/peft](https://github.com/AlanAnsell/peft) | Dynamic scatter sparse | RigL/SM3 기반 prune-regrow selection으로 sparse delta 위치를 재선택하며 학습 |
| **SMT** | [HectorHHZ/Sparse_Matrix_Tuning](https://github.com/HectorHHZ/Sparse_Matrix_Tuning) | Block-sparse (256x256) | calibration gradient score 상위 256x256 block을 선택하여 학습 |
| **S2FT** | [Infini-AI-Lab/S2FT](https://github.com/Infini-AI-Lab/S2FT) | Structured heads/channels | random 또는 score 기반으로 attention head와 FFN channel을 선택하고, coupled row/column co-permutation 후 dense submatrix를 학습 |
| **LT-SFT** | [cambridgeltl/composable-sft](https://github.com/cambridgeltl/composable-sft) | Lottery ticket | dense mask-search 후 `|theta_search - theta_0|` 상위 element를 선택하여 sparse fine-tuning |

모든 method에서 **~170M trainable parameters** (전체 7B 모델의 ~2.35%)를 사용합니다.

## Architecture

```
sparse_ft/
├── methods/           # 각 method의 원본 코드 (수정 포함)
│   ├── sift/         # SIFT - SparseLinear + index cache
│   ├── spiel/        # SpiEL - AlanAnsell peft fork
│   ├── smt/          # SMT - BlockSparseLinear
│   ├── s2ft/         # S2FT - structured head/channel selection
│   └── ltsft/        # LT-SFT - SparseLinear (lottery ticket)
├── pipeline/          # LMFlow 통합 파이프라인
│   ├── __init__.py
│   ├── sift_tuner.py  # SIFT: frozen weight(buffer) + sparse_delta(Parameter)
│   ├── spiel_tuner.py # SpiEL: peft fork SftConfig + merge_and_unload
│   ├── smt_tuner.py   # SMT: BlockSparseLinear (256x256 blocks)
│   ├── s2ft_tuner.py  # S2FT: structured attention-head / FFN-channel selection
│   ├── ltsft_tuner.py # LT-SFT: lottery-ticket diff top-k selection
│   ├── train_method.py      # 통합 학습 entrypoint
│   ├── eval_mtbench.py      # MT-Bench 평가 (vLLM + GPT-4o-mini judge)
│   ├── eval_lmharness.py    # MMLU/CSR 평가 (lm-eval-harness + vLLM)
│   ├── prepare_datasets.py  # 데이터셋 변환
│   └── verify_merge.py      # Merge 검증
├── configs/
│   ├── ds_zero1.json         # DeepSpeed ZeRO-1 기본
│   └── ds_zero1_sift.json    # DeepSpeed ZeRO-1 + optimizer/scheduler (sparse method용)
├── run_all.sh         # 전체 실험 파이프라인 (MT-Bench → MMLU → CSR)
├── run_mtbench.sh     # MT-Bench only
└── smoke_test.sh      # 각 method smoke test (20 steps)
```

## Key Design: DeepSpeed 8-GPU Compatible Sparse Training

모든 sparse method가 DeepSpeed ZeRO-1 + 8 GPU에서 동작하도록 통일된 패턴 적용:

```python
class SparseLinear(nn.Module):
    """Drop-in replacement for nn.Linear with sparse trainable delta."""
    def __init__(self, orig_linear, flat_idx):
        super().__init__()
        # Frozen weight → buffer (DeepSpeed가 optimizer에서 제외)
        self.register_buffer("weight", orig_linear.weight.data)
        self.register_buffer("flat_idx", flat_idx)
        # Trainable sparse delta → Parameter (DeepSpeed가 optimize)
        self.sparse_delta = nn.Parameter(torch.zeros(len(flat_idx), ...))
    
    def forward(self, x):
        delta_flat = torch.zeros(self.weight.numel(), ...)
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        return F.linear(x, self.weight + delta_flat.view(self.weight.shape), self.bias)
```

**핵심**: `weight`를 `register_buffer`로 등록해서 DeepSpeed가 gradient buffer를 할당하지 않음 → OOM 방지.

## Setup

```bash
# conda 환경, LMFlow, PEFT fork, sparse_ft 연결을 한 번에 설정
cd /home/nksol0405/LLM/sparse_ft
bash scripts/setup_conda_env.sh
```

## Running Experiments

### Smoke Test (각 method 동작 확인, 20 steps)
```bash
cd /home/nksol0405/LLM/sparse_ft

# 개별 method
bash smoke_test.sh sift
bash smoke_test.sh spiel
bash smoke_test.sh smt
bash smoke_test.sh s2ft
bash smoke_test.sh ltsft
```

### Full Training + Evaluation
```bash
# 전체 실험 (MT-Bench → MMLU → CSR)
cd /home/nksol0405/LLM/sparse_ft
bash run_all.sh

# 진행상황 확인
tail -f /data/nksol0405/LLM/rapa/results/results.md
```

### Individual Training
```bash
# GPU 작업은 스크립트가 srun으로 감싼 뒤 deepspeed를 실행한다.
cd /home/nksol0405/LLM/sparse_ft
METHODS=sift bash run_mmlu_20m_single_gpu.sh
```

## Hyperparameters

### MT-Bench Task
| Param | Value |
|-------|-------|
| Model | `mistralai/Mistral-7B-v0.3` |
| Dataset | `timdettmers/openassistant-guanaco` (oasst1) |
| Batch size/GPU | 1 |
| Grad accum | 1 |
| LR scheduler | linear |
| Learning rate | 5e-5 |
| Epoch | 1 |
| Max seq length | 512 |
| # Trainable params | 170M |
| Judge | GPT-4o-mini |

### MMLU (5-shot) / CSR (0-shot)
| Param | Value |
|-------|-------|
| Model | `meta-llama/Llama-2-7b-hf` |
| Dataset | `/data/nksol0405/LLM/rapa/OwLore_Dataset/mmlu/mmlu.json`, `CSR_DATASET` |
| LR scheduler | cosine |
| Learning rate | 5e-5 |
| Epoch | 1 |
| Max seq length | 512 |
| # Trainable params | 170M |

## SIFT Index Caching

SIFT sparse index 생성은 7B 모델에서 시간이 걸립니다. 최초 1회 생성 후 `.pt` 파일로 캐시:
- 캐시 위치: `/data/nksol0405/LLM/rapa/checkpoints/.sift_idx_cache/`
- 캐시 키: `hash(model_name + sparse_rate + modules + seed + dataset_name)`
- 같은 모델 + 같은 설정이면 데이터셋이 달라도 `torch.load`로 즉시 로드

## Important Notes

1. **OOM 방지**: 모든 데이터(모델, 캐시, 체크포인트)를 `/data/nksol0405/LLM/rapa/`에 저장
2. **GPU 사용**: 현재 A100 서버에서는 기본적으로 `srun` 없이 실행하며, 단일 GPU는 `GPU_INDEX=0` 또는 `GPU_INDEX=1`로 선택
3. **DeepSpeed config**: `ds_zero1_sift.json`에 optimizer + scheduler 명시 필요 (frozen params와의 호환성)
4. **SpiEL fork**: `AlanAnsell/peft` fork 필요, `linear_sd` C extension 빌드 필수
