# Multiple Choice Evaluation（選擇題評測）

> 涵蓋 `pattern`、`box`、`custom_regex` 三種評測方法，適用於 MMLU、MMLU-Pro、TMMLU+ 等選擇題 benchmark。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | 選擇題（MMLU / MMLU-Pro / TMMLU+ 等） |
| **evaluation_method** | `box`（預設推薦）/ `pattern` / `custom_regex` |
| **實作狀態** | ✅ 完整實作（專案核心功能） |
| **需要 optional deps** | 不需要 |
| **實作者** | Twinkle AI Team |

---

## 1. 來源

### MMLU

- **標題**：Measuring Massive Multitask Language Understanding
- **作者**：Dan Hendrycks et al.
- **發表**：ICLR 2021
- **連結**：https://arxiv.org/abs/2009.03300
- **HuggingFace**：`cais/mmlu`

### MMLU-Pro

- **標題**：MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **作者**：Yubo Wang et al.
- **發表**：NeurIPS 2024
- **連結**：https://arxiv.org/abs/2406.01574
- **HuggingFace**：`TIGER-Lab/MMLU-Pro`

### TMMLU+

- **標題**：TMMLU+: An Improved Traditional Chinese Evaluation Suite
- **作者**：iKala
- **HuggingFace**：`ikala/tmmluplus`

### 本專案 Example 資料

| 資料集 | 路徑 | 筆數 |
|--------|------|------|
| MMLU | `datasets/example/mmlu/` | 20 |
| MMLU-Pro | `datasets/example/mmlu_pro/` | 20 |
| TMMLU+ | `datasets/example/tmmluplus/` | 20 |

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

選擇題評測衡量模型的知識廣度與推理能力。模型需從給定選項中選出正確答案。
本專案透過選項隨機排列（`shuffle_options`）消除模型對選項位置的偏好。

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | 正確答案的比例 | ✅ |

---

## 3. Leaderboard

- **MMLU Leaderboard**：https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
- **Open LLM Leaderboard**：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

---

## 4. 本專案實作說明

### 三種 Extractor

| 方法 | Extractor | 說明 |
|------|-----------|------|
| `pattern` | `PatternExtractor` | 正則表達式匹配答案字母（含中英文模式） |
| `box` | `BoxExtractor` | 提取 `\boxed{X}` 中的答案（推薦） |
| `custom_regex` | `CustomRegexExtractor` | 使用者自訂正則表達式 |

### Scorer

所有選擇題方法共用 `ExactMatchScorer`：正規化後精確比對答案字母。

### 特殊設計決策

- **`box` 模式推薦**：搭配 system prompt 指示模型使用 `\boxed{}` 格式，提取穩定度最高
- **選項隨機排列**：`shuffle_options: true` 可在每次評測時隨機排列選項順序，消除位置偏好
- **動態選項偵測**：不硬編碼 A/B/C/D，支援任意數量的選項（如 MMLU-Pro 的 A–J）

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/mmlu/"
  evaluation_method: "box"
  system_prompt:
    en: "Please read the question carefully and select the best answer. Present your final answer in the format \\boxed{answer}, e.g., \\boxed{A}."
  shuffle_options: true
```

### 完整 config template

參見 `twinkle_eval/config.multiple_choice.template.yaml`
