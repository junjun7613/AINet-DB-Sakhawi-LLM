# 訓練データと抽出結果のサンプル

[訓練の概要](../../TRAINING_OVERVIEW.md)で説明した訓練データと、モデルの抽出結果から、少数の例を抜き出したものです。データ全体ではありません。

| ファイル | 件数 | 内容 |
|---|---|---|
| [train_sample.jsonl](train_sample.jsonl) | 20 | 訓練データ(v3)の例。学習に使った形式そのままです |
| [test_extractions.jsonl](test_extractions.jsonl) | 20 | test の例。本文・正解・2 つのモデルの出力・採点の内訳 |
| [test_extractions.md](test_extractions.md) | 20 | 上と同じ 20 件を、Web で読みやすい表にしたもの |

## 選び方

- **訓練データ**: v3 の train(3,330 件)から、関係のあるエントリ 15 件と関係のないエントリ 5 件を無作為に選びました。
- **抽出結果**: 旧データでも test だった 319 件から選びました。学習前のモデルの出力があるのはこの範囲だけだからです。関係のあるエントリ 16 件(概要 2.5 節の例 AIND-D02212 を含む)と、関係のないエントリ 4 件です。
- 無作為抽出の乱数の種は固定しています。実際のデータでは、約 4 割のエントリに関係がありません。

## `train_sample.jsonl` の形式

1 行に 1 エントリの JSON Lines です。形式は概要の 2.4 節のとおりです。

```json
{"id": "AIND-D…", "messages": [
  {"role": "system", "content": "指示文(英語、全エントリ共通)"},
  {"role": "user", "content": "伝記本文(アラビア語)"},
  {"role": "assistant", "content": "{\"relations\": [...], \"offices\": [...]}"}]}
```

学習では `system` と `user` を入力とし、`assistant` の JSON を出力するように学習します。

## `test_extractions.jsonl` の形式

1 行に 1 エントリです。

| フィールド | 内容 |
|---|---|
| `id` | エントリの ID |
| `text` | 伝記本文(モデルへの入力) |
| `gold` | 正解(v3)。`relations` と `offices` のリスト |
| `gold_not_scored` | 正解のうち、人名が本文に現れないため採点から外した関係 |
| `zero_shot` | Gemma 4 12B(追加学習なし)の結果 |
| `finetuned_v3` | Gemma 4 12B を v3 データで追加学習したモデルの結果 |

`zero_shot` と `finetuned_v3` は、それぞれ次のフィールドを持ちます。

| フィールド | 内容 |
|---|---|
| `parse_ok` | 出力が JSON として読めたか。`false` のとき `output` は `null` |
| `output` | モデルの出力 |
| `score` | 指標ごとの採点。`tp` = 正しかった件数、`fp` = 余計に出した件数、`fn` = 取りこぼした件数 |
| `excluded_from_scoring` | 採点から外したモデルの出力(人名が本文に現れない人物、系譜の父) |

`score` の指標(`rel_strict`、`rel_lenient` など)の意味は、概要の 4 章を参照してください。

---

*Samples (20 entries each) of the v3 training data and of model outputs on the test set (gold, zero-shot Gemma 4 12B, and Gemma 4 12B fine-tuned on v3, with per-metric scores). See the [English overview](../../TRAINING_OVERVIEW.en.md), Sections 2.4 and 4, for the format and metrics.*
