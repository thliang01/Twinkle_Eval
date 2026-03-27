# Logit Evaluation（Logit-based 評測）

> 使用 `logit` 評測方法，透過 completions API 的 log probability 評估模型對各選項的偏好。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | Logit-based 選擇題評測 |
| **evaluation_method** | `logit` |
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | 不需要（但需要 API 端點支援 completions API + echo 模式） |
| **實作者** | Twinkle AI Team |

---

## 1. 來源

Logit-based 評測是一種通用的選擇題評測方法，參考自 lm-evaluation-harness 的設計。
不綁定特定 benchmark，可用於任何選擇題資料集。

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

與 `box` / `pattern` 模式不同，logit 模式不要求模型生成文字答案，
而是透過 completions API 計算每個選項的 log probability，選擇機率最高的選項作為答案。

### 優勢

- 不受模型輸出格式影響（不需要模型理解 `\boxed{}` 格式）
- 更接近模型內部的知識表達
- 每個問題需要 N 次 API 呼叫（N = 選項數），但本專案透過並行請求加速

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | log probability 最高的選項為正確答案的比例 | ✅ |

---

## 3. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/logit.py
```

`LogitExtractor`（`uses_logprobs = True`）：
- 對每個選項呼叫 `LLM.score_continuation(context, option)`
- 取 log probability 最高的選項作為答案
- 每個問題 N 次 API 呼叫，全部在 ThreadPoolExecutor 中並行

### Scorer

共用 `ExactMatchScorer`。

### API 需求

API 端點必須支援 completions API（非 chat completions）的 `echo=True` 模式。
目前已知支援的服務：vLLM、text-generation-inference。
OpenAI 官方 API 不支援 echo 模式。

---

## 4. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/mmlu/"
  evaluation_method: "logit"
```

> 注意：使用 logit 模式前，請確認您的 API 端點支援 completions API + echo 模式。
