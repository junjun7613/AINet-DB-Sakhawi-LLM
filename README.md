# AINet-DB-Sakhawi-LLM

このリポジトリは、**アラビア語人名録を対象としたローカル LLM の訓練**に関する資料を保存・共有するためのものです。訓練のデータや、学習・評価の概要を置きます。

対象は、アル＝サハーウィー(al-Sakhāwī, 902 H / 1497 CE 没)の人物録『輝く光(*al-Ḍawʾ al-Lāmiʿ*)』です。AINet-DB プロジェクトで校閲した構造化データ(TEI XML)を正解として使います。手元の計算機で動く公開モデルを追加学習し、伝記本文から人間関係や官職などの情報を抽出させます。

## 現在公開している資料

現時点では、訓練の概要と、訓練データ・抽出結果のサンプルを公開しています。

| ファイル | 内容 |
|---|---|
| [TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md) | 訓練の概要。データの作り方、学習・評価の方法、結果、今後の予定を説明しています |
| [TRAINING_OVERVIEW.en.md](TRAINING_OVERVIEW.en.md) | 訓練の概要の英語版 |
| [figures/](figures/) | 概要で使っている図(英語版の図は `*_en.svg`) |
| [data/sample/](data/sample/) | 訓練データと抽出結果のサンプル(各 20 件)。[抽出結果の一覧](data/sample/test_extractions.md)で、正解とモデルの出力を見比べられます |

訓練用のデータ全体などは、準備が整いしだい追加します。

---

*This repository stores and shares materials on training local LLMs for an Arabic biographical dictionary (al-Sakhāwī's* al-Ḍawʾ al-Lāmiʿ*). At present it contains the training overview and samples of the training data and model outputs ([data/sample/](data/sample/)). Overview: [TRAINING_OVERVIEW.en.md](TRAINING_OVERVIEW.en.md) (English) or [TRAINING_OVERVIEW.md](TRAINING_OVERVIEW.md) (Japanese).*
