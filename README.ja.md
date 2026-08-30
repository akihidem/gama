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
| **`ABMCTSBackend`** | **探索**：ノード毎に**幅を広げる**（新規候補）か**深く掘る**（既存を改良）かを Thompson サンプリングで決める。報酬＝外部検証。幅を広げる時にどのモデルを呼ぶかも bandit（Multi-LLM AB-MCTS） |

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

### AB-MCTS ── *適応探索*としての構造
振り分け・合議・meshflow は組み合わせの「形」を**事前に**決める。`ABMCTSBackend` はその形を推論時に
適応させる。探索木を育てながら、**どのノードでも**次の 1 コールを**幅を広げる**（まったく新しい候補を
生成）か**深く掘る**（既存の候補を改良）かを Thompson サンプリングで選ぶ ── 報酬は外部の
`verify`→score。各 worker が 1 つの bandit *アクション*なので、幅を広げる手番では*どの*モデルを呼ぶかも
オンライン学習される＝小型モデルの束がそのまま **Multi-LLM AB-MCTS** になる。実装は Sakana AI
*"Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search"*
([arXiv:2503.04412](https://arxiv.org/abs/2503.04412)・NeurIPS 2025、blog
[sakana.ai/ab-mcts](https://sakana.ai/ab-mcts/)、OSS [SakanaAI/treequest](https://github.com/SakanaAI/treequest)）
の閉形式版 **AB-MCTS-A** を stdlib のみで移植したもの。`verify` スコアが既に `[0,1]` なので Beta 共役の
Thompson サンプリングは `random.betavariate` だけで済み、numpy/scipy/PyMC は要らない:
```bash
gama run "<task>" --config examples/abmcts.example.json --task-type code_implementation
gama bench --backends abmcts,ensemble,ssh-openai --config recipes/mac-studio-abmcts/config.json --tier large
```
「深く掘る」は親の回答**とそのスコア**を改良プロンプトに戻す ── これが深さを幅（＝ただの best-of-N）に
潰さない唯一の勘所。正直な検証は「探索が単体モデルに勝つか」ではなく「*適応的な*幅/深さ探索が、同じ
モデル群の*幅だけ*の best-of-N に勝つか」なので、対照として `ensemble` を並べて回し
（`recipes/mac-studio-abmcts` 参照）、`last_trace` で勝ったノードが本当に depth > 1 で `refine` されて
いるか確認すること。移植の線引きは `trinity.py` と同じ正直さで、実装したのは AB-MCTS-A であり、MCMC
ベースの AB-MCTS-M（PyMC が必要で stdlib-only を壊す）ではない。

### 構造が*割に合うか*を測る ── `bench --suite hard` / `market` / `mesh`
合議や段階委譲が効くのは特定の条件下だけ。gama は推測でなく**自分のモデルで測れる**ようにする。
既定の bench suite は天井効果（良いモデルは全部 1.0 で判別不能）なので、まず判別 suite に切り替える:
```bash
gama bench --backends ollama,ssh-openai --suite hard        # or: --suite brutal, --suite wide
```
`hard` / `brutal` は*難しく*する suite で、**`wide`** は*広く*する suite（40 問・5 クラス各 8 問・
難易度は `hard` と同じ帯）。深さは 2 つの backend を順位づけるのに要り、広さは suite を
search / confirm / sealed に**割る**のに要る（後述の `gama grow`。26 問しか無いと confirm が
5 問になり、1 問が得点の 0.2 を動かしてしまう）。全 suite の正解は
`tests/test_suite_integrity.py` が参照解から再導出している。答えが間違っている case は、
正解したモデルを黙って減点するため。

`wide` の実測（2026-08-18・WSL2・CPU のみの ollama。箱が違えば変わるし、CI は
この数字を守っていない）: `llama3.2:3b` **0.550** / `qwen2.5:7b` **0.725** /
`qwen2.5-coder:7b` **0.817**。能力の順に並び、天井にも当たっていない。割って使う前に suite が
満たすべき条件はここ。40 問のうち 19 問がこの 3 つを判別し、17 問は 3 つとも解ける
（無駄ではなく、易しいクラスを*壊す*変異を捕まえる役に立つ）、4 問はどれも解けない。

**`--suite steep` は、他の suite を既に飽和させたモデル向け。** `gemma4:e2b` は他5 suite を
まとめたプールで 0.95 を取り、封印 20 問に対する伸びしろが 1 問しか残らない。そこでは
`gama grow` は何も昇格させられず、止めているのはループではなく天井の方になる。`steep` は
厳密な冪剰余・階乗の末尾ゼロ・CSV のエスケープ・LRU の退避順など 20 問で、**7B が頭で解くには
難しく、プログラムなら大半が自明**。構造が効く余地を残してある。実測: `llama3.2:3b` **0.468** /
`qwen2.5:7b` **0.683** / `gemma4:e2b` **0.894**。正直な限界として、gemma4 は 20 問中 16 問を
完答しており、これは天井を上げただけで取り払ってはいない。

**`--suite graded` だけは分数で採点する。** 他の suite は実質 pass/fail なので、スコアは
case 単位でしか動かない。graded の 20 問は 1 問が独立に検証できる要件を複数持ち（小問 3 つ、
書式制約 4 つ、CSV セル 6 つなど）、満たした割合を返す。決定的なまま、判定モデルも無しで、
粒だけ細かくする。これが要る理由は 2 つ: `gama grow` の「confirm 1 問ぶん」の床が固定定数と
違う結果を出すのは、実際の改善が**半問と 1 問の間**に落ちた時だけで、pass/fail ではそこに
落ちようがない。もう 1 つは `abmcts` のような推論時探索が verify スコアを reward にして
進むため、二値だと「深く掘る」枝が磨く先を持たない。同じ箱での実測で、測定の
**35% / 22% / 15%**（3B / 7B / 7B-coder）が 0 と 1 の厳密な間に落ちた（観測値 0.33, 0.43,
0.5, 0.6, 0.67, 0.75, 0.8。1/2 刻みではなく 1/5・1/7 も出る）。正直な注記として、graded は
`wide` より**易しい**（平均 0.73 / 0.90 / 0.94）ので、勾配を得る用途に使い、強い backend の
判別には使わない。
**`gama market` ── いつ「束ねる」が「大きくする」より安いか？** 検証エスカレーション（meshflow）が
単体最強を Pareto 支配するのは、**安いティアの完全解率 `p` がコスト比 `w/s` を超えるとき**（`p > w/s`）。
`gama market` はあなたのティア（安→高）で bench を回し、コスト・正答率・支配の verdict を出す:
```bash
gama market --backends gemma,haiku --suite hard --costs 1,10
```
**`gama mesh` ── 「合議」は本当に効くか？** アンサンブルが単体最強を超えるのは、**メンバが脱相関
（`rho < 1`）かつ相互相補（入れ子でない）なときだけ** ── `利得 = (1−rho)·(1−p)·(1−(1−p)^(n−1))`。
`gama mesh` は失敗相関 `rho` と union−best 利得を bench から測り、デプロイ*前*に「点火するか・ただ
トークンを燃やすだけか」を教える:
```bash
gama mesh --backends gemma,qwen,llama --suite hard
```
両者は上の合成 backend に対する*経済/統計の verdict 層*で、[`soshiki-genron`](https://github.com/akihidem/soshiki-genron)（組織原論）研究repo（`model/market.py` の p>w/s・`model/mesh.py` の脱相関）から移植。「規模でなく構造」を標語から**自分のハードで反証できる**ものに変える。

## 結果
hard 12 問・Mac Studio(MLX) で全部ローカル。測定を公平化済（コード抽出＋トークン予算）──
これは*互角*で、クリーンな勝ちではない:

| | 主権的軽量スタック（7B+24B+32B+tool・振り分け） | 単体 122B |
|---|---|---|
| スコア | **0.92** | **0.92** |
| 取りこぼし | r4（曜日 mod 演算） | c3（roman 数字のコード） |

穴は相補的で同点 ── しかも全部ローカル。再現:
`python3 -m experiments.moa_vs_strong <config.json>`。

### 自分で育つ ── `gama grow`
`bench` は*あなたが書いた*組み合わせを測る。**`grow` は組み合わせの方を書く**。config を 1 手ずつ
変異させ（クラスを別モデルへ振る、レーンを `tool` で包む、統合役を立てて合議にする、検証 gate の
段階委譲にする、未ルーティングの全クラスが乗る既定レーンを差し替える、合成の中にもう一段合成を
入れる、そして**構造を剥がして素に戻す**）、同じ決定的チェッカで全候補を測り、
**held-out split が確認したときだけ**チャンピオンを差し替える。ループのどこにも判定役の LLM は
居ない。

ループ自身の変異も測る価値があり、実際に 1 つ壊れているのが見つかった。提案される ensemble は
`majority` を使っていたが、これは返答を逐語比較するため、自由文では全部が同数になり、
集約結果が黙って**最初のメンバー**になる。graded suite で **0.705 —— 素の 3B の 0.830 を下回り、
レイテンシは 14 倍**。「2 つのモデルに金を払って安い方の答えを採用する」変異になっていた。
現在の提案はもう一方のメンバーに統合させる（同じ case で 0.975・20 問中 8 勝 0 敗）。

```bash
gama grow --models llama3.2:3b,qwen2.5:7b,qwen2.5-coder:7b --generations 4 --width 5 \
          --out grow.jsonl --write-recipe recipes/my-box
```
既定で `wide,hard,brutal`（56 問）をプールし、3 つに割る:

| split | そこで決めてよいこと |
|---|---|
| `search` | 候補を測り、世代ごとに**1 本だけ**挑戦者を選ぶ（K 個の最大値は上振れするので、これは挑戦権であって昇格の根拠ではない） |
| `confirm` | 昇格を決める唯一の場。**max（confirm 1 問ぶん, チャンピオン自身の測り直しの揺れ）以上**の差を要求する（1 問に満たない差も、自分の揺れより小さい差も、改善ではない） |
| `sealed` | 何も決めない。走り終わるまで一度も触らず、最後に 1 回だけ開ける。だから看板の数字だけは、どの判定にも当てはめていない |

WSL2（CPU のみ ollama）で 12 走まわし（最大 86 問・43/23/20 分割）、構造的な変更が 2 つ見つかった。
`qa` を tool レーンへ通すことと、`research` を外部検証つきで 7B coder へ段階委譲すること。
それぞれが担当するクラスで測ると、`qa` は **0.222 → 1.000**（プールのどのモデルも素では解けない
問題を含む。3 時間 45 分を 15300 秒と答えるが、同じ 3B に「プログラムで」と言うと `3*3600+45*60`
と書く）、`research` は **0.313 → 0.583**。

ここに至るまでに公開済みの結論を 2 回撤回していて、2 回目の方が学びが大きい。
`research → meshflow` は**追加**としては 9 回中 3 回しか昇格せず、だから一度はレシピから外した。
ところがそれを含むチャンピオンを種にした走行で外そうとすると、削除は confirm を 0.142 下げ、
ノイズ幅 0.0217 を大きく超えた。**昇格記録は「その日のチャンピオンに勝てたか」を測っていて、
「入っている状態で何を支えているか」は測っていない。** 2 つの問いは食い違い、レシピが答えるべき
なのは後者。いまは両レーンを載せ、12 走ぶんの記録も一緒に置いてある:
[`recipes/grown-wsl-ollama`](recipes/grown-wsl-ollama)。

別のマシンの 48B（AWS L4 上で llama.cpp が配信する `Kimi-Linear-48B-A3B`・SSH 経由）に向けると、
ループはまず `tool:<クラス>` 変異を 5 本とも却下した（どれも confirm を下げた）。この README は
それを「必要としないモデルへの税」と書いた。同じ 80 問を 20 / **40** / 20 に割り直す（昇格の床が
0.025 に半減する）と、`tool:qa`（+0.031）と `tool:research`（+0.031）が昇格し、封印スコアは
0.833 → 0.850 へ動いた。

どちらの測定も正確で（drift は終始 0.0000）、食い違っているのは計器ではなく集計の方。
**効果は case ごとに向きが違い、大きさは 40 問中 1 問ぶん。それより粗い分割は、引きしだいで
ゼロとも害とも報告する。** 本プロジェクトが撤回した結論はこれで 3 つ目で、3 つとも出どころは
同じ「1 つの分割からの断定」。

ここから引きたくなる教訓にも、算数の側から 1 つ訂正が要る。**「case を増やせばよい」は一般には
成り立たない。** 床は `1/n`、利得は `S/n`（S = 実際に改善した case 相当数）で、**両方が同じだけ
縮むので比は S のまま**。変異が触らないクラスの case を足しても 1 ミリも動かない。要求は
`S >= 1`、つまり「**丸 1 問ぶん動いたか**」であって、効くレバーは**変えているクラスの case を
増やすこと**。ループは利得を分数でなく case 単位で報告するようにした。

モデルプールを丸ごと入れ替えると（`gemma4:e2b` + `qwen2.5:7b`・素から）、5 世代で昇格ゼロだった。
種が最初から search 0.959 で、**その suite はそのモデルに対して飽和している**から。この走行は
「構造が移植できるか」には答えられない代わりに、もっと鋭いことを言う。同じ封印 20 問で、
素の 7.2GB モデルが **0.950（10.85s/問）**、構造化した 3B が **0.865（2.02s/問）**。
構造が買ったのは*安いモデルが使い物になること*であって、天井の高さではない。レイテンシを
払えるなら、構造より規模の方が簡単。それを数字にするために `gama market` がある。

却下がループの仕事の大半で、しかも門のあらゆる部分から出る。挑戦権を得られなかった候補、confirm
の改善は本物だがその世代のチャンピオン自身の揺れより小さかった候補（+0.054 対 敷居 0.071）、
split が粗すぎて証明できなかった候補。台帳にはそれも、**最有力の変更が一度も提案されないまま
終わった走行**（探索幅が狭く別候補がキュー先頭に居座った）も残してある。台帳上、不在は却下と
同じ顔をする。台帳を残す理由がそこ。

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
- 同一モデルのコピーを重ねても **無意味** ── 効くのは多様性（別々の blind spot）。`gama mesh` は
  これを失敗相関 `rho` で定量化する: 同一/冗長メンバ → `rho ≈ 1` → 不点火。
- 全メンバーが共有する穴は小型アンサンブルでは塞げない ── 共有する hard core は `rho` が高く、
  `gama mesh` は `gain 0` を返す。そこは道具か、大きいモデルが要る。
- 異種アーキの比較は公平な答え抽出＋十分なトークンが要る。さもないとモデルでなくハーネスを
  測ってしまう。
- `tool` とコードのベンチケースは **モデル生成 Python を実行** する ── 信頼できる backend で
  のみ（opt-in・サンドボックス的）。

## ライセンス
MIT。[`tehai`](https://github.com/akihidem/tehai-core) 委譲レイヤーから、焦点を絞った単体
ツールとして切り出した。
