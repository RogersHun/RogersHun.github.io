---
title: "PReQ 추론 최적화 Part 1: Profiling에서 FX Graph rewrite와 AMP까지"
description: "PReQ 배포 모델에 Lookup 재구성, Auto-Tuning, INT8, torch.compile, FX rewrite, SQLite 캐시, BF16 AMP를 적용하고 품질과 CPU 지연을 비교한 실험 기록"
date: 2026-10-20 00:00:00 +0900
categories: [AI, Inference Optimization]
tags: [pytorch, profiling, quantization, torch-compile, torch-fx, auto-tuning, sqlite, mixed-precision]
pin: false
math: false
mermaid: false
---

기존 연구에서 모델을 압축한 뒤, 실제 배포 모델의 추론 과정에는 어떤 개선 여지가 있는지 궁금해졌다. 저장 공간이 작아져도 실행이 빨라진다는 보장은 없기 때문이다.

PReQ 배포 체크포인트를 기준으로 CPU 추론을 측정하고, Lookup 실행 구조·설정·정밀도·컴파일 경로를 바꿔 보았다. 기존 라이브러리 적용에서 한 단계 더 나아가 작은 FX Graph rewrite pass와 측정 이력 Database도 구현했다. 이 글에서는 그 과정을 **Part 1**으로 정리한다.

결과가 모두 속도 향상으로 이어지지는 않았다. INT8·BF16과 직접 작성한 rewrite는 해당 실행에서 원본보다 느렸다. 반면 실행 설정 탐색과 TorchInductor 적용에서는 소폭의 개선을 관찰했다. 무엇을 적용했는지와 함께, 실제 연산이 바뀌었는지·추천 결과가 유지됐는지·어떤 후보를 제외했는지를 기록했다.

## 1. 대상 모델과 측정 범위

PReQ는 압축된 텍스트·이미지 표현을 추천에 활용한다. 이번 배포 모델의 계산 흐름을 단순화하면 다음과 같다.

```text
RQ code + folded codebook
    → Lookup · stage 합산
    → 사용자/아이템별 GatedFusion
    → ID embedding과 혼합
    → 3-layer LightGCN sparse propagation
    → 최종 embedding 캐시
    → 사용자·아이템 점수 계산과 Top-K 추천
```

`folded codebook`은 학습된 projection을 미리 반영한 기존 배포 형식이다. 이 folding 자체를 이번 실험에서 새로 개발한 것은 아니다. 사용자 5,753명, 아이템 9,090개, graph embedding 64차원의 동일한 체크포인트를 사용했다. Part 1에서는 재학습하지 않았다.

추론 시간을 두 종류로 구분했다.

- **Embedding refresh:** 모든 사용자·아이템의 최종 embedding을 다시 계산하는 시간.
- **Cached request:** 이미 계산한 embedding으로 점수와 Top-K를 구하는 시간. 기본 요청 batch는 사용자 32명이다.

전체 embedding을 매 요청마다 갱신하는 서비스와, 갱신 결과를 캐시해 여러 요청에 사용하는 서비스는 비용 구조가 다르다. 따라서 refresh 시간의 개선을 그대로 온라인 요청 지연 개선으로 해석하지 않았다.

### 비교할 때 고정한 것과 주의할 것

비교는 각 Colab 실행 안에서 같은 모델과 split을 사용해 수행했다. Profiler와 dtype 확인 hook은 반복 지연 측정에서 분리했고, 여러 설정의 실행 순서를 섞었다. 탐색 이후 별도 확인 측정을 두는 실험도 구성했다.

**서로 다른 노트북 실행의 절대 시간을 이어 붙여 speedup을 계산하지 않았다.** Colab의 CPU와 공유 부하는 달라질 수 있다. 아래 각 표는 해당 실험 안의 비교이며, 서로 다른 단계의 개선율을 합산할 수 없다.

품질은 같은 history mask와 `torch.topk` evaluator로 비교했다. 이 evaluator는 논문의 evaluator와 동일하지 않으며, 원본 입력 특징이 train 데이터만으로 생성됐는지도 이번 추론 실험에서 별도로 검증하지 않았다. 따라서 여기의 NDCG·Recall은 구현 변경의 영향을 살피는 비교 지표로 사용한다. 새로운 추천 정확도 성과를 주장하는 실험은 아니다.

## 2. Part 1 실험 구성

| 단계 | 질문 | 구현·측정한 내용 |
|---|---|---|
| 01 Baseline | 어디에 시간이 드는가? | 연산·구간별 Profiling, 정상 실행 지연, 메모리 관찰 |
| 02 Lookup | 반복 Lookup을 묶으면 달라지는가? | sequential / batched_ordered / batched_reduce 비교 |
| 03 Auto-Tuning | 실행 설정 중 무엇이 유리한가? | mode·chunk·CPU thread 조합 탐색과 확인 측정 |
| 04 INT8 | 일부 Linear를 양자화하면 유리한가? | Projection·Gate별 Dynamic Quantization |
| 05 Compiler | 기존 컴파일러를 적용하면 달라지는가? | eager / aot_eager / TorchInductor 비교 |
| 06 FX rewrite | 그래프를 직접 바꿔 볼 수 있는가? | 병렬 Linear를 packed batched GEMM으로 변환 |
| 07 Database | 이미 측정한 설정을 다시 측정해야 하는가? | SQLite 이력·index·캐시 무효화·재검증 |
| 08 AMP | 낮은 정밀도를 어디에 적용할 것인가? | BF16 autocast와 제한된 정책 자동 선택 |

## 3. 01 — Baseline Profiling

처음에는 모델 변경 없이 측정 코드를 추가했다. 기준 모델과 구간 이름을 추가한 모델의 최대 출력 오차는 0이었다.

해당 실행에서 전체 refresh 평균은 약 **48.98 ms**, cached request 평균은 약 **1.31 ms**였다. 모델 tensor의 논리적 크기는 약 **5.98 MiB**, 최종 embedding 캐시는 약 **3.62 MiB**였다.

이 크기는 프로세스의 최대 RAM이 아니다. 그래프 자료구조, 중간 tensor, Python 객체, allocator 등은 별도로 봐야 한다. Profiler에 표시되는 순할당량도 peak memory와 다르다.

Profiler에서는 Lookup과 Fusion, sparse graph propagation에 비용이 분포했다. 특히 반복적인 `embedding`, `clone`, `contiguous` 호출을 관찰해 Lookup 실행 구조를 다음 실험 대상으로 정했다. Profiler가 들어간 구간 시간은 계측 비용을 포함하므로 정상 실행의 refresh 시간과 직접 합산하지 않았다.

## 4. 02 — Lookup을 묶어서 실행하기

기존 코드는 RQ stage마다 텍스트와 이미지 codebook을 각각 조회하고 더했다. 여러 작은 호출을 한 번에 처리할 수 있도록 stage별 codebook을 펼치고 index에 offset을 더했다.

```python
# 개념을 보여 주기 위한 축약 코드
# codes: [batch, stages], books: [stages, codewords, dim]
offset = torch.arange(stages, device=codes.device) * codewords
flat_books = books.reshape(stages * codewords, dim)
looked_up = F.embedding(codes + offset, flat_books)
features = looked_up.sum(dim=1)
```

비교한 경로는 세 가지다.

- `sequential`: stage별 Lookup과 합산.
- `batched_ordered`: Lookup은 묶되 합산 순서는 유지.
- `batched_reduce`: Lookup을 묶고 `sum`으로 stage를 합산.

전체 refresh Profiler에서 `embedding` 호출은 **240회 → 30회**, `clone`·`contiguous`는 각각 **240회 → 0회**로 줄었다. 추천 Top-K는 5,753명 모두 동일했다.

하지만 호출 감소가 곧 전체 가속을 뜻하지는 않았다. `batched_reduce`의 refresh paired median speedup은 약 **1.039배**였고, 다른 요약 지표와 tail에는 변동이 컸다. `batched_ordered`는 약 0.957배로 오히려 느렸다. 이 단계만으로 안정적인 전체 모델 가속을 주장하지 않았다.

묶어서 조회한 텍스트·이미지 tensor 두 개의 논리적 크기는 4 MiB였다. 호출을 줄이는 대신 중간 tensor가 생기는 trade-off도 기록했다. 이 값 역시 프로세스 최대 메모리를 뜻하지 않는다.

## 5. 03 — 실행 설정 Auto-Tuning

Lookup 방식만 바꾸는 대신 다음 조합을 탐색했다.

| 항목 | 후보 |
|---|---|
| Lookup | sequential, batched_reduce |
| chunk | 256, 1024, 4096 |
| CPU threads | 1, 2 |

출력·추천 결과 검증을 통과한 설정만 측정하고, 탐색에서 고른 후보는 새로운 confirmation block에서 다시 비교했다. 채택 규칙에는 paired speedup, block별 승리 횟수, 평균과 P95를 사용했다. 이는 작은 유한 격자 탐색이며 DP나 RL은 아니다.

해당 실행에서는 `batched_reduce / chunk 1024 / threads 1`이 선택됐다.

| Confirmation | 기준 설정 | 선택 설정 |
|---|---:|---:|
| 평균 refresh | 75.94 ms | 69.11 ms |
| P95 refresh | 100.55 ms | 81.57 ms |

평균은 약 **9.0%**, P95는 약 **18.9%** 감소했고, 다섯 confirmation block에서 모두 기준보다 빨랐다. 기준은 `sequential / chunk 1024 / threads 2`였다. Lookup 방식과 thread 수가 함께 바뀌었으므로 개선 전체를 Lookup 하나의 효과로 돌릴 수 없다.

탐색 중 baseline anchor의 큰 변화도 관찰했다. 독립된 머신·시간대까지 포함한 성능 보장은 아니며, 이후 실험에서는 설정을 고정해 다른 변경의 영향을 비교했다.

## 6. 04 — 선택적 INT8 Dynamic Quantization

PReQ에서 이미 사용하는 RQ code와, 이번에 Linear에 적용한 INT8 Quantization은 서로 다른 대상이다. 기존 code·codebook을 바꾸지 않고 Fusion 안의 Linear만 선택적으로 변환했다.

- Projection Linear 4개만 INT8.
- Gate Linear 4개만 INT8.
- 위 Linear 8개 모두 INT8.
- FP32 기준.

x86 backend에서 실제 `quantized::linear_dynamic` 호출을 확인했다. 변환된 state를 저장·재로딩한 출력도 검증했다.

| 정책 | 평균 refresh | 체크포인트 크기 |
|---|---:|---:|
| FP32 | 35.02 ms | 5.983 MiB |
| Projection INT8 | 37.41 ms | 5.939 MiB |
| Gate INT8 | 36.54 ms | 5.939 MiB |
| 전체 Fusion Linear INT8 | 43.31 ms | 5.895 MiB |

세 정책 모두 미리 정한 **Validation NDCG 상대 하락 1% 이내** 기준을 통과했다. 그러나 전체 refresh는 빨라지지 않아 FP32를 유지했다. 품질 기준을 통과했다고 추천 목록까지 완전히 같은 것은 아니다.

원본 모델 tensor 약 5.98 MiB 중 FP32 Linear weight는 약 0.125 MiB, 약 2.1%였다. 일부 Linear만 압축했기 때문에 전체 체크포인트 감소도 최대 약 **1.48%**였다. 작은 행렬과 동적 변환 비용은 속도 저하의 원인 후보지만, 이 실험만으로 원인을 확정하지 않았다.

여기서는 기존 `torch.ao.quantization.quantize_dynamic` 호환 API를 사용했다. 실행 당시 deprecation 경고도 기록했으며, 후속 구현에서는 PyTorch의 torchao 이전 안내를 확인할 필요가 있다.

## 7. 05 — TorchDynamo·TorchInductor 적용

다음으로 dense 연산이 모여 있는 사용자·아이템의 `GatedFusion`만 `torch.compile`로 감쌌다. Lookup, sparse LightGCN, 점수 GEMM은 eager FP32로 유지했다.

```python
compiled_fusion = torch.compile(
    fusion,
    backend="inductor",
    fullgraph=True,
    dynamic=True,
)
```

`fullgraph=True`는 지정한 Fusion 모듈 내부에서 그래프가 끊기면 실패하도록 하는 설정이다. 전체 모델에 Graph break가 없다는 뜻은 아니다.

실험에서는 backend를 감싼 기록 함수로 실제 전달된 FX 그래프를 저장했다. `torch._dynamo.explain`의 진단용 그래프와 실제 backend 호출 기록을 구분했다.

| 경로 | 평균 refresh |
|---|---:|
| eager | 62.93 ms |
| aot_eager | 67.55 ms |
| TorchInductor | 60.20 ms |

TorchInductor는 해당 실행에서 평균 refresh가 약 **4.3% 감소**했고, 세 측정 block에서 모두 eager보다 빨랐다. refresh와 요청을 함께 실행한 경우는 약 3.2% 감소했다.

사용자·아이템 Fusion은 각각 graph count 1, Graph break 0이었다. 반복 지연 측정 중 새 backend 그래프는 없었다. valid/test 각각 5,753명의 Top-K 순서·집합과 NDCG·Recall은 동일했고, 최대 embedding 오차는 약 `1.34e-7`이었다.

최초 실행은 컴파일을 포함해 약 **27.9초** 걸렸다. 이는 반복 refresh 약 60 ms와 다른 비용이다. 짧게 한 번만 실행하는 상황과 계속 재사용하는 상황에서 판단이 달라질 수 있다.

이 단계는 TorchDynamo·TorchInductor 활용 및 실행 분석 경험이다. 컴파일러 내부를 직접 개발하거나 특정 low-level fusion을 구현한 것은 아니다.

## 8. 06 — FX Graph rewrite pass 직접 구현

컴파일러 적용에서 더 나아가, 그래프를 직접 바꾸는 작은 pass를 구현했다. 대상 패턴은 GatedFusion의 병렬 Projection 두 개다.

```text
변환 전: proj_t(t), proj_v(v)
변환 후: stack(t, v) → bmm → bias 추가 → 두 출력 선택
```

학습된 두 weight를 한 번 미리 묶고, 추론 시 두 입력을 stack해 batched GEMM으로 처리한다. 일반적인 단일 행렬에 block-diagonal weight를 만들어 불필요한 곱셈을 추가하는 방식은 사용하지 않았다.

아래 그림은 실험 코드에서 추출한 실제 FX node와 의존 관계다. 주황색 두 Projection이 초록색 packed 모듈로 바뀐다.

<div style="display:flex;gap:12px;align-items:flex-start;">
  <div style="width:50%;"><strong>변환 전</strong><img src="{{ '/assets/img/preq-part1/fu_before.svg' | relative_url }}" alt="FX rewrite 전: 별도 proj_t와 proj_v" style="width:100%;height:auto;" loading="lazy"></div>
  <div style="width:50%;"><strong>변환 후</strong><img src="{{ '/assets/img/preq-part1/fu_after.svg' | relative_url }}" alt="FX rewrite 후: packed_projection과 두 출력 선택" style="width:100%;height:auto;" loading="lazy"></div>
</div>

하나의 박스가 하나의 하드웨어 커널을 뜻하지는 않는다. packed 모듈 내부를 펼치면 입력 stack·bmm·bias·출력 선택이 여전히 존재한다.

<details>
<summary>packed 모듈 내부 FX 그래프 펼쳐 보기</summary>
<img src="{{ '/assets/img/preq-part1/fu_packed_expanded.svg' | relative_url }}" alt="packed 모듈 내부의 stack, bmm, bias와 출력 선택" style="max-width:100%;height:auto;" loading="lazy">
</details>

### Pass 구현에서 확인한 것

1. `proj_t`, `proj_v`가 각각 한 번 등장하는지 검사.
2. 두 Linear의 weight 크기·dtype·device와 입력 의존성 검사.
3. packed 모듈과 두 출력 선택 node 삽입.
4. 기존 node 사용처를 바꾸고 두 Linear node 제거.
5. `graph.lint()`로 그래프 연결 검사, `recompile()`로 실행 코드 갱신.
6. 여러 batch·마지막 chunk 및 전체 모델 출력 검증.

지원 범위는 같은 크기의 2D 입력을 받는 고정 가중치 추론 Projection이다. 지원하지 않는 패턴은 거절한다. 역전파나 임의의 Linear 조합을 처리하는 범용 pass는 아니다.

Profiler에서 실제 실행 구조가 바뀐 것을 확인했다.

| 연산 호출 | eager | FX traced | FX packed |
|---|---:|---:|---:|
| `aten::linear` | 60 | 60 | 30 |
| `aten::bmm` | 0 | 0 | 15 |
| `aten::stack` | 0 | 0 | 15 |

Colab 실행에서 출력 비교 표의 embedding·score 오차는 0이었고, valid/test Top-K와 NDCG·Recall도 동일했다. 다만 속도는 다음과 같았다.

| 경로 | 평균 refresh |
|---|---:|
| eager | 87.04 ms |
| FX traced | 108.12 ms |
| FX packed | 98.08 ms |

packed는 traced보다 평균이 짧았지만, 원본 eager보다는 약 **12.7% 느렸다**. 입력 stack과 메모리 이동 등 추가 비용이 있으므로 Linear 호출 수만으로 가속을 판단할 수 없었다. 측정 변동도 있어 원인을 확정하지 않았다.

직접 구현한 것은 **병렬 Projection의 실행 구조를 바꾸는 제한된 FX transformation**이다. Quantization operation fusion이나 CUDA·NEON 커널 구현과는 구분한다.

## 9. 07 — SQLite 기반 Auto-Tuning 측정 이력

같은 설정을 계속 재측정하는 비용을 줄이기 위해 SQLite Database를 구성했다.

| 테이블 | 내용 |
|---|---|
| contexts | checkpoint·split·코드 hash, 환경, 측정 protocol |
| configs | mode·chunk·CPU threads |
| searches | 탐색 실행과 baseline anchor |
| validations | 설정별 새 품질 검증 |
| measurements | 완료된 측정 기록과 시간 |
| samples | block·repeat별 실제 지연 |
| cache_accesses | hit/miss·이유·참조한 측정 |

같은 모델·설정만으로는 캐시를 허용하지 않았다. CPU 정보, PyTorch build, affinity와 runtime session, 코드 hash, TTL도 확인한다. 현재 baseline을 짧게 재측정한 **anchor**가 과거와 크게 다르면 저장된 지연을 사용하지 않는다. cache hit여도 출력과 Validation 추천은 새로 검증한다.

```sql
CREATE INDEX idx_measurement_cache
ON measurements(context_key, config_key, created DESC);

SELECT *
FROM measurements
WHERE context_key = ? AND config_key = ? AND status = 'complete'
ORDER BY created DESC
LIMIT 1;
```

실제 `EXPLAIN QUERY PLAN`에서도 위 index 사용을 확인했다. 측정과 전체 샘플은 transaction으로 함께 저장했다. SQLite 작업은 로컬 파일에서 하고, Drive에는 닫힌 snapshot을 저장하는 단일 writer 방식으로 구성했다.

### Colab에서는 캐시가 왜 전부 miss였는가?

| 항목 | initial search | cached repeat |
|---|---:|---:|
| cache hit | 0 / 12 | 0 / 12 |
| 새 지연 샘플 | 180 | 180 |
| 생략한 지연 샘플 | 0 | 0 |
| baseline anchor | 153.12 ms | 68.83 ms |
| 탐색 소요 시간 | 68.07 s | 54.83 s |

반복 탐색의 miss 이유는 12개 모두 **`anchor_drift`**였다. anchor가 약 2.22배 차이 나서 허용 기준 1.25배를 넘었다. 캐시 조회가 고장 난 것이 아니라, 과거 결과 재사용을 거절하는 분기가 실행됐다.

따라서 **68.07초 → 54.83초 감소를 캐시 효과로 보고하지 않았다.** 두 번 모두 180개 지연 샘플을 새로 측정했다. DB에는 측정 24개·샘플 360개가 저장됐고, 최종 후보도 fresh confirmation 기준을 통과하지 못해 baseline을 유지했다.

로컬 기능 검증에서는 캐시 hit와 중복 측정 생략도 확인했지만, 이를 Colab에서 얻은 성과처럼 합쳐 쓰지는 않았다. 캐시가 제안하는 설정은 과거 측정에 따른 후보일 뿐이며, 최종 선택에는 새로운 측정이 필요하다.

## 10. 08 — BF16 AMP와 정밀도 정책 자동 선택

마지막으로 `torch.autocast`를 이용해 Projection과 Gate의 계산 정밀도를 바꿨다. INT8 실험과 달리 **저장 가중치는 FP32로 유지**한다.

| 정책 | Projection | Gate |
|---|---|---|
| fp32 | FP32 | FP32 |
| projection_bf16 | BF16 autocast | FP32 |
| gate_bf16 | FP32 | BF16 autocast |
| fusion_linear_bf16 | BF16 autocast | BF16 autocast |

Normalization, sigmoid, 혼합, codebook, ID embedding, sparse propagation과 점수 GEMM은 FP32로 유지했다. Linear 뒤에 FP32로 돌아오는 경계를 명시했다.

```python
# Projection 구간의 축약 예시
with torch.autocast("cpu", dtype=torch.bfloat16, enabled=projection_amp):
    t = self.proj_t(t)
    v = self.proj_v(v)

t = F.normalize(t.float(), dim=-1)
v = F.normalize(v.float(), dim=-1)
```

출력 hook으로 실제 dtype을 확인하고, 반복 측정 전에 hook을 제거했다. Projection 정책은 Linear 4개, Gate 정책은 4개, 전체 정책은 8개에서 BF16 출력을 확인했다. 최종 embedding은 FP32였다.

### 자동 선택의 기준

1. Validation NDCG 상대 감소 1% 이내인 정책을 허용.
2. 허용된 정책 중 탐색 지연으로 후보 제안.
3. BF16 후보가 제안되면 FP32와 fresh confirmation.
4. speedup·평균·P95·block별 승리 조건을 통과해야 채택.
5. 선택을 마친 뒤 Test를 보고하고 정책 JSON을 저장·복원 검증.

네 개의 정해진 정책을 비교하는 작은 탐색기다. 새로운 범용 AMP 알고리즘을 제안한 것은 아니다.

| 정책 | 평균 refresh | FP32 대비 소요 시간 |
|---|---:|---:|
| FP32 | 78.69 ms | 1.00배 |
| Projection BF16 | 173.32 ms | 2.20배 |
| Gate BF16 | 148.58 ms | 1.89배 |
| 전체 Fusion Linear BF16 | 234.10 ms | 2.98배 |

Validation NDCG 상대 감소는 Projection 약 **0.027%**, Gate 0%, 전체 약 **0.019%**로 모두 기준을 통과했다. 다만 추천 순서가 바뀐 사용자는 각각 786명·90명·795명이었고, 추천 집합이 바뀐 사용자는 129명·20명·129명이었다. 품질 기준 통과와 추천 목록 동일성은 다른 조건이다.

탐색에서 FP32가 가장 빨라 **`keep_fp32`**가 선택됐다. 이 실행에서는 BF16 후보가 제안되지 않았기 때문에 confirmation은 FP32만 재측정했다. Test 표 역시 선택된 FP32의 결과이며, BF16 Test 성능을 보여 주는 표가 아니다.

작은 행렬, CPU 지원, autocast context와 캐스팅 비용 등이 속도 저하의 원인 후보다. dtype 확인만으로 native BF16 하드웨어 가속 여부를 단정하지 않았다. 이번 구현의 구간별 context·변환 비용은 측정에 포함됐다.

## 11. Part 1에서 얻은 구현 경험

Part 1에서는 다음을 실제 모델에 적용하고 검증했다.

- PyTorch Profiler 기반 연산·구간 분석과 측정 코드 구성.
- Lookup batching과 실행 설정 유한 탐색.
- 선택적 INT8 Dynamic Quantization과 BF16 AMP 비교.
- TorchDynamo·TorchInductor 활용 및 그래프 포착 진단.
- 작은 `torch.fx` Graph rewrite pass 구현·시각화.
- SQLite schema·index·transaction·측정 이력 캐시와 무효화.
- Validation 품질 제약과 실측 지연에 따른 제한된 정밀도 정책 선택.

각 변경이 빨라질 것이라고 가정하는 대신, 실제 연산·출력·추천 결과와 정상 실행 지연을 따로 확인했다. 다음 Part 2에서는 이 기반 위에서 추가 최적화 문제와 하드웨어 실행 범위를 확장할 예정이다. DP/RL, 직접 CUDA·NEON 커널, Multi-device allocation은 이번 Part 1에서 구현한 항목에 포함하지 않는다.

## 12. 결과 기록과 재현 시 확인할 항목

본문 표의 Colab 평균·P95를 모은 [요약 CSV]({{ '/assets/data/preq-part1/reported_results.csv' | relative_url }})를 함께 남겼다. 이 CSV는 실행 출력에서 옮긴 **집계값**이며 raw latency sample 전체가 아니다.

재현 시에는 단순한 노트북 이름보다 다음 조건을 함께 남겨야 한다.

- 원본 checkpoint·split hash와 입력 형식.
- Python·PyTorch 버전과 CPU·thread 설정.
- Lookup 방식·chunk·정밀도 적용 범위.
- warmup·반복 횟수·실행 순서·확인 측정 여부.
- evaluator와 history mask, Top-K 변화·품질 기준.
- Profiler 포함 여부와 평균·P50·P95의 구분.

이 글의 수치는 2026-10-20에 실행한 실험 기록이다. 같은 코드라도 공유 CPU 부하와 라이브러리 구현에 따라 결과가 달라질 수 있다. 배포 판단에는 대상 환경에서 다시 측정한 결과가 필요하다.

## 참고 자료

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization.html)
- [Python sqlite3](https://docs.python.org/3/library/sqlite3.html)
