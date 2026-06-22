```text
                                      ___
              .-"""-.   .-"""-.      (   )  ~
             /   o   \ /   o   \      )_(    puff
            |     >   V   <     |    /|\     (kiseru)
             \     '-...-'     /    / |
          _.'-------------------'-._/
         /         G A M A          \
        |          '--www--'         |
         \     croak ... croak      /
          '._                    _.'
             '-..____________..-'
```

> **口寄せ！** 小さなローカルモデルを — 振り分け・合議・道具で — 束ねて、大きいモデルと
> 戦う蝦蟇。（*gama* = 蝦蟇。NARUTO のガマブンタのように口寄せする蛙。）

[English](README.md) | **日本語**

# gama 🐸 — ローカル LLM を組み合わせる

[![CI](https://github.com/akihidem/gama/actions/workflows/ci.yml/badge.svg)](https://github.com/akihidem/gama/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![deps: stdlib only](https://img.shields.io/badge/deps-stdlib%20only-brightgreen.svg)](pyproject.toml)

**各タスクを得意な小型ローカルモデルへ振り分け、複数を束ね（mixture of agents）、道具を
持たせ、どの組み合わせが大きいモデルに並ぶかをベンチで測る。stdlib only。全部ローカル。**

> **きっかけの発見**: 難問スイートで、小型ローカルモデルの*構造化された組み合わせ*（7B +
> 24B + 32B ＋ 電卓ツールをタスク別に振り分け）が **単体の 122B と同点（0.92 vs 0.92）** ──
> 1 台の Mac だけ・クラウドなし。コピーを重ねても（無意味）、素朴に束ねても（0.83）ダメで、
> **各タスククラスを正しい軽量機構へ振る**ことで並んだ。規模でなく構造。

`gama` は、その組み合わせを*あなたの*ハードで作って測るツール。そして「**どの小型モデル＋
道具＋振り分けが大きいモデルに並ぶか**（どのハードで）」を持ち寄って育てる場。

## なぜ
暗算できない小型モデルも `print(...)` を書いて実行すれば解ける。ある種の推論が弱いモデルも
アンサンブルで多数決にかければ救える。コード特化モデルは汎用モデルにコードで勝つ。**タスク別
に正しい小型の専門家を組み合わせれば、大きいモデルに並べる** ── ローカル・主権的・安価に。

## インストール
```bash
pip install git+https://github.com/akihidem/gama        # または: pip install gama-llm
# 開発するなら:
git clone https://github.com/akihidem/gama && cd gama && pip install -e .
```
依存ゼロ ── 純 Python ≥ 3.10。

## 30 秒クイックスタート
gama は OpenAI 互換のローカルサーバ（**ollama**・**MLX `mlx_lm.server`**・**LM Studio**・
**vLLM**）と subprocess CLI を叩く。
```bash
# 無料・決定的スモーク（モデル不要）:
gama bench --backends echo

# 自分のローカルモデルをクラス別に測り、振り分け表を提案:
gama bench --backends ollama --tier large --propose routing.json
```

## 部品
| backend | 役割 |
|---|---|
| `ollama`, `ssh-openai` | ローカルモデルを呼ぶ（HTTP、または SSH 越しの OpenAI サーバ＝ポート非開放） |
| **`GamaBackend`** | タスククラスで **振り分け** 1→1（実測の `routing_table`） |
| **`EnsembleBackend`** | 同一タスクに N モデルを **合議**（`synthesize` / `majority` / `first`） |
| **`ToolBackend`** | **道具(PAL)**：モデルに Python を書かせて実行（正確な計算など） |
| **`MeshflowBackend`** | **段階委譲**：外部検証で gate した安→強エスカレーション＋縁で合議＋高stakesは人間膜（AIネイティブの*組織の形*） |

JSON で自由に合成（`build_backend`）：`tool` / `ensemble` / コーダーの上に `gama` ルータを
乗せた*主権的スタック*を、単体の大きいモデルとベンチで比べられる。
```bash
gama bench --backends gama,ssh-openai --config recipes/mac-studio-mlx/config.json --tier large
```

### meshflow ── *組織*としての構造
振り分け・合議はモデルを*静的に*束ねる。`MeshflowBackend` は欠けていた「形」＝**検証エスカレーション**
を足す。まず一番安いティアを試し、**外部の `verify(artifact)→score` が通ったときだけ**採用（モデルの
自己申告でなく）。通らなければ強いティアへ昇格。どの単独ティアも通らない**縁**では試行を**合議**（誤りが
相補的だから効く）。なお未解決で stakes が高ければ黙って ship せず `<<NEEDS_HUMAN>>` を返す＝薄い人間
統治膜。こうして**普段は安いティアで済ませ、検証が要求したときだけ強いティアに届く**。
```bash
gama run "<task>" --config examples/meshflow.example.json --task-type code_implementation
gama bench --backends meshflow,ssh-openai --config examples/meshflow.example.json --tier large
```
これは「規模でなく構造」を*組織*の実行系にしたもの ──
[`soshiki-genron`](https://github.com/akihidem/soshiki-genron)（組織原論）研究repo
（`experiments/meshflow.py`・PAPER §6.5「採用すべき組織図」）で第一原理から導かれ、frontier モデルに
低コストで並ぶことが示された形を、gama に移植した。

## 結果
hard 12 問・Mac Studio(MLX) で全部ローカル。測定を公平化済（コード抽出＋トークン予算）──
これは*互角*で、クリーンな勝ちではない:

| | 主権的軽量スタック（7B+24B+32B+tool・振り分け） | 単体 122B |
|---|---|---|
| スコア | **0.92** | **0.92** |
| 取りこぼし | r4（曜日 mod 演算） | c3（roman 数字のコード） |

穴は相補的で同点 ── しかも全部ローカル。再現:
`python3 -m experiments.moa_vs_strong <config.json>`。

## レシピ ── みなで育てる 🌱
`recipes/` はコミュニティ・ライブラリ。1 レシピ = `config.json`（組み合わせ）＋ `recipe.md`
（モデル群・ハード・`gama bench` の数値）。あなたの箱で大きいモデルに並ぶ小型の組み合わせを
見つけたら、**レシピを追加** ── [CONTRIBUTING](CONTRIBUTING.md) 参照。
```bash
gama recipes                       # 一覧
gama recipes mac-studio-mlx        # レシピの config を表示
gama run "47*53+89*17 を計算" --config recipes/mac-studio-mlx/config.json --task-type qa
```

## 正直な注意
- 同一モデルのコピーを重ねても **無意味** ── 効くのは多様性（別々の blind spot）。
- 全メンバーが共有する穴は小型アンサンブルでは塞げない ── そこは道具か、大きいモデルが要る。
- 異種アーキの比較は公平な答え抽出＋十分なトークンが要る。さもないとモデルでなくハーネスを
  測ってしまう。
- `tool` とコードのベンチケースは **モデル生成 Python を実行** する ── 信頼できる backend で
  のみ（opt-in・サンドボックス的）。

## ライセンス
MIT。[`tehai`](https://github.com/akihidem/tehai-core) 委譲レイヤーから、焦点を絞った単体
ツールとして切り出した。
