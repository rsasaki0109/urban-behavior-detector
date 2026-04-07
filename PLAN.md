# Urban Behavior Detector — 実装計画・引き継ぎノート

> **この文書の目的**: 次の実装担当（Claude 等）がリポジトリの現状・意図・未完了を短時間で把握できるようにする。  
> **リポジトリ絶対パス**: `/workspace/ai_coding_ws/urban-behavior-detector`  
> **親ワークスペース規約**: `/workspace/ai_coding_ws/AGENTS.md`（スマホ調査・セキュリティ検証用途のメタ規約。本repoは都市行動検知だが、**機微情報をコミットしない**ことは共通）

---

## Claude / 次エージェント向け引き継ぎ（最初に読む）

### いま動いているもの

- **パイプライン**: YOLO 検知 → SORT 追跡 → 複数 `BehaviorAnalyzer` → JSON + アノテ動画 + イベント用スナップショット（`pipelines/video_pipeline.py`）。
- **入力**: 動画ファイル・RTSP・ウェブカメラ（`scripts/run_stream.py`）。
- **違反タイプ**: `walking_smoking`, `bicycle_phone`, `bicycle_umbrella`, `bicycle_wrong_way`, `signal_violation`, `sidewalk_riding`, `walking_phone`（README の表が正）。
- **テスト**: `pytest` で **67 件**（2026-04 時点）。CI は `ruff` + `pytest`（`.github/workflows/ci.yml`）。

### 実装の「主軸」と「補助」

- **主軸（本番向き）**: 幾何・追跡・ルールベース。再現性があり説明しやすい。
- **補助**: `scripts/vlm_evaluate.py` がイベント JSON とスナップショットを読み、VLM に投げて `vlm_evaluation` を付与。**Ollama / OpenAI 互換バックエンド**対応済み（`--backend openai --base-url http://host:8000` で vLLM 等に接続可能）。レート制御・リトライ・タイムアウトも統一済み。パイプライン本体とは切り離された後処理。

### 既知の限界（README Dogfooding と一致）

- **遠景・低解像・小物体**: タバコが数 px 級だと **cigarette YOLO も誤検知**（カバン・マスク等）。本番では **1080p・近景・取り付け高**の前提が README に明記されている。
- **Jetson + VLM**: Cosmos Reason 2B は Jetson 上で動作するが、物理空間推論特化で **タバコ・スマホ等の小物体認識は弱い**。**Qwen2.5-VL-3B のほうが汎用 visual QA に強くこのユースケースに適合**。全フレーム VLM は遅延・メモリ的に厳しく、**イベント駆動（スナップショットのみ）**が現実的。`vlm_evaluate.py --backend openai` で vLLM 経由接続可能。

### 触るべき主要ファイル（優先度順）

| 目的 | パス |
|------|------|
| パイプライン統合 | `pipelines/video_pipeline.py` |
| 設定の真実 | `configs/*.yaml`（`rules.yaml` 既定、`rules_cigarette.yaml` でタバコモデル） |
| 歩きタバコ | `behaviors/walking_smoking.py`, `detectors/cigarette_detector.py`, `detectors/pose_detector.py` |
| VLM 後処理 | `scripts/vlm_evaluate.py`（Ollama / OpenAI 互換） |
| ROI キャリブレーション | `scripts/calibrate_roi.py`（信号・歩道ゾーンを GUI で描画） |
| CLI | `scripts/run_demo.py`, `scripts/run_stream.py` |
| ドキュメント | `README.md`（ユーザー向け）、本 `PLAN.md`（開発引き継ぎ） |

### 推奨オンボーディング手順（15–30 分）

1. `README.md` の Quick start と Output format を読む。
2. `pytest tests/ -q` と `ruff check .` を通す（環境があれば）。
3. `python scripts/run_demo.py` でサンプル動画 1 本（`samples/` または手元 mp4）。
4. `walking_smoking` の `update()` 分岐（タバコ検出と pose 振動の AND/OR）を `behaviors/walking_smoking.py` で確認。
5. 次に取り組むタスクを下記 **Phase 8+** から 1 件選び、スコープを明文化してからコードを触る。

### ユーザーからの明示要求（会話メモ）

- **Jetson / VLM**: `vlm_evaluate.py` の **OpenAI 互換バックエンド対応は実装済み**（`--backend openai --base-url http://jetson:8000`）。Cosmos Reason 2B は動くが小物体認識が弱いため **Qwen2.5-VL-3B を推奨**。主判定を VLM のみに寄せるのは非推奨（ブレ・遅延・説明責任）。

---

## プロジェクト概要

ライブカメラ映像や動画から都市における迷惑行為・交通違反っぽい行動を検知する OSS。**GitHub Pages** でデモを見せ、コードより先に体験できる構成。  
2026 年 4 月施行の **自転車の青切符**関連違反を README でマーケ文言として整理済み（金額・类型は法改正の追随が必要なら README 側を更新）。

---

## 完了フェーズ一覧（履歴）

### Phase 1: コア実装 [完了]

- [x] ディレクトリ構成
- [x] `detectors/yolo_detector.py` — YOLOv8 ラッパー
- [x] `trackers/sort_tracker.py` — IoU ベース SORT、速度・方向
- [x] `behaviors/base.py` — `BehaviorAnalyzer`, `ViolationEvent`, `compute_confidence()`
- [x] `behaviors/walking_smoking.py` — 歩きタバコ（後述のとおり **pose + オプション cigarette** に発展）
- [x] `behaviors/bicycle_violation.py` — 自転車ながらスマホ・傘
- [x] `pipelines/video_pipeline.py`
- [x] `scripts/run_demo.py`, `scripts/export_demo_assets.py`
- [x] `configs/rules.yaml`, `requirements.txt`

### Phase 2: デモサイト [完了]

- [x] `docs/index.html`, `docs/assets/css/style.css`, `docs/assets/js/app.js`
- [x] サンプル JSON・デモインデックス
- [x] `README.md`

### Phase 3: 実動画検証 [完了]

- [x] CC ライセンス動画での `run_demo` 検証、アセット再エンコード、`export_demo_assets`

### Phase 4: 精度改善 [完了]

- [x] 歩きタバコ: YOLOv8-pose による手首–鼻距離の時系列（振動 vs スマホの「張り付き」）
- [x] 自転車–人物ペアリング改善
- [x] `compute_confidence()` 共通化

### Phase 5: 追加違反タイプ [完了]

- [x] `behaviors/wrong_way.py`
- [x] `behaviors/signal_violation.py` + `detectors/signal_detector.py`
- [x] `behaviors/sidewalk_riding.py`

### Phase 6: デモサイト強化 [完了]

- [x] 動画プレーヤー・イベント連動・フィルタ・統計・OGP・ヒートマップ

### Phase 7: プロダクト品質 [完了]

- [x] pytest（現行 **67** テスト）
- [x] GitHub Actions（ruff + pytest）
- [x] MIT `LICENSE`, `.gitignore`, Docker（`Dockerfile`）
- [x] ストリーミング（`scripts/run_stream.py`）

### Phase 7 以降に追加済み（PLAN 追記）

- [x] **Cigarette 専用検出**: `detectors/cigarette_detector.py`, 学習済み重みは `models/`（git 管理方針は `.gitignore` 確認）
- [x] **歩きスマホ**: `behaviors/walking_phone.py` + テスト `tests/test_walking_phone.py`
- [x] **VLM 評価スクリプト**: `scripts/vlm_evaluate.py`（Ollama API）
- [x] 複数プリセット設定: `configs/rules_*.yaml`
- [x] 補助スクリプト: `scripts/gradio_demo.py`, `scripts/extract_event_frames.py`, `scripts/generate_heatmap.py`

---

### Phase 8: 運用・品質強化 [完了]

- [x] `vlm_evaluate.py` の **バックエンド抽象化**（Ollama / OpenAI 互換 `--backend` / `--base-url`）
- [x] **レート制御・リトライ・タイムアウト**の統一（`--rate-limit`, `--timeout`, 指数バックオフ）
- [x] **ROI 追従（人物 bbox クロップ）**パイプラインオプション（`detection.roi_crop` で遠景精度改善）
- [x] **ROI キャリブレーション GUI**（`scripts/calibrate_roi.py` — 信号・歩道ゾーンをマウスで描画 → YAML 出力）
- [x] **メモリリーク修正**（Track 履歴上限、stale track pruning、OpenCV リソース finally 化、`_reported` set 上限）
- [x] **構造化ログ（JSON Lines）**オプション（`--log-jsonl` で `pipeline_start` / `violation_detected` / `pipeline_stop` をリアルタイム書き出し）

---

## Phase 9+（未完了・候補バックログ）

優先度はプロダクト方針次第。実装する場合は **Issue 化せずとも本節にチェックボックス**で良いが、完了時に `[x]` へ更新すること。

### A. エッジ / VLM

- [ ] Jetson 上 **Qwen2.5-VL-3B**（推奨）または Cosmos Reason 2B の **実測レイテンシ・メモリ**記録

### B. 検知品質

- [ ] `walking_smoking`: 現状 **振動のみでは発火しない**（`has_cigarette` なしの `has_oscillation` だけでは `smoking_detected` にならない）。意図的だが、**pose のみ環境**でのフォールバックポリシーは要プロダクト判断

### C. 運用・信頼性

- [ ] Prometheus / メトリクスは需要が出てから

### D. 法務・倫理（プロダクトが公共カメラを触る場合）

- 撮影・告知・保存期間・個人特定可能性は **リーガルレビュー対象**。コード外だがリリースノートや README に注意書きを足すかはステークホルダー判断。

---

## アーキテクチャ詳細

### パイプライン処理フロー（現行）

```
入力（ファイル / RTSP / カメラ）
  │
  ▼
YOLODetector.detect(frame)  → Detection[]（人物・車両・phone 等）
  │
  ├─（任意）PoseDetector → キーポイント
  ├─（任意）CigaretteDetector → タバコ候補 bbox
  │
  ▼
SORTTracker.update(detections) → Track[]（track_id, speed, direction, …）
  │
  ▼
各 BehaviorAnalyzer.update(frame_idx, tracks, detections, pose?, cigarettes?)
  │  （signal は ROI 上書き画像など追加引数あり — 実装を参照）
  ▼
ViolationEvent[] → マージ・スナップショット保存 → JSON + 動画
```

`scripts/vlm_evaluate.py` はこの **後**に動き、JSON 内の `snapshot` を読んで API 呼び出し。

### 違反判定ロジック（現行コードに合わせた要約）

#### 歩きタバコ (`walking_smoking`)

**前提**: `person` かつ `track.speed >= speed_threshold`（歩行中）。

1. **Pose（任意）**: 各人に対応する pose を IoU でマッチ。手首–鼻距離 / 身長 の時系列から **近接↔離間の振動**を検出（スマホは「常に近い」になりやすい）。
2. **Cigarette detector（任意）**: `detection.cigarette_model` が設定されているとき、bbox が人物上半身付近に重なるか。
3. **発火条件（実装どおり）**  
   - `has_cigarette and has_oscillation` → 検知（高めの confidence 補正）  
   - `has_cigarette` のみ → 検知（やや低補正）  
   - **振動のみ** → `smoking_detected` にならない（誤検知抑制のため）

**設定**: `configs/rules.yaml` の `walking_smoking` と `detection.use_pose` / `cigarette_model`。

#### 自転車ながらスマホ / 傘 (`bicycle_violation`)

- cyclist 認定（person–bicycle の空間・速度整合）の上で、phone または umbrella の幾何条件＋持続フレーム。

#### 逆走 (`wrong_way`)

- 追跡方向と期待方位の角度差。

#### 信号無視 (`signal_violation`)

- ROI の信号色（HSV）と **横断ゾーン**内進入の組み合わせ（README: 付近歳行者の誤爆を抑える狙い）。

#### 歩道走行 (`sidewalk_riding`)

- ROI ポリゴン（歩道）内での自転車。

#### 歩きスマホ (`walking_phone`)

- phone 物体＋顔付近＋ pose パターン（テスト: `tests/test_walking_phone.py`）。

### ファイル構成と責務（2026-04 時点）

```
urban-behavior-detector/
├── detectors/
│   ├── yolo_detector.py
│   ├── pose_detector.py
│   ├── cigarette_detector.py
│   └── signal_detector.py
├── trackers/
│   └── sort_tracker.py
├── behaviors/
│   ├── base.py
│   ├── walking_smoking.py
│   ├── walking_phone.py
│   ├── bicycle_violation.py
│   ├── wrong_way.py
│   ├── signal_violation.py
│   └── sidewalk_riding.py
├── pipelines/
│   └── video_pipeline.py
├── configs/
│   ├── rules.yaml
│   ├── rules_cigarette.yaml
│   ├── rules_full.yaml
│   ├── rules_pose_only.yaml
│   ├── rules_pose.yaml
│   ├── rules_sidewalk.yaml
│   ├── rules_traffic.yaml
│   └── rules_wrongway.yaml
├── scripts/
│   ├── run_demo.py
│   ├── run_stream.py
│   ├── vlm_evaluate.py         # Ollama / OpenAI 互換 VLM 評価
│   ├── calibrate_roi.py        # ROI ポリゴン GUI キャリブレーション
│   ├── export_demo_assets.py
│   ├── generate_heatmap.py
│   ├── gradio_demo.py
│   └── extract_event_frames.py
├── models/                    # cigarette 等（リポジトリに含まれない場合あり）
├── tests/                     # pytest 67
├── docs/                      # GitHub Pages
├── Dockerfile
├── .github/workflows/ci.yml
├── README.md
├── PLAN.md                    # 本ファイル
└── requirements.txt
```

### `configs/rules.yaml` の考え方

- **トップレベルキーごと**に `enabled` と閾値を持つ設計が基本。
- `detection` に **グローバルなモデル・信頼度・クラス一覧・pose/cigarette 切替**。
- 現場ごとに YAML を複製し、`signal_rois` / `crossing_zones` / `sidewalk` ポリゴンを調整する想定。

---

## VLM 連携（`vlm_evaluate.py`）

### 現状

- **バックエンド**: Ollama（既定）と **OpenAI 互換**（vLLM, LiteLLM 等）の 2 系統
  - `--backend ollama` → `http://localhost:11434/api/generate`
  - `--backend openai --base-url http://host:8000` → `/v1/chat/completions`
- **レート制御**: `--rate-limit N`（秒間 N リクエスト）、リトライ最大 3 回（指数バックオフ）、`--timeout` 設定可能
- **プロンプト**: 英語。出力は JSON 形式を期待（`smoking_detected`, `confidence`, `description`）
- **入力**: イベント JSON 内のスナップショットパス
- **推奨モデル**: Jetson では **Qwen2.5-VL-3B**（汎用 visual QA に強い）。Cosmos Reason 2B は物理空間推論特化で小物体認識が弱い。

---

## 設計判断の記録（なぜそうしたか）

### SORT（ByteTrack ではない）

- MVP として実装コストと依存（numpy のみ）のバランス。
- `Track` インタフェースを保てば差し替え可能。

### 歩きタバコで「振動のみ」を採用しない

- 遠景・遮蔽・習癖で **手の振動だけ**だと誤検知が増えるため、`cigarette` または明確なオブジェクト証拠を優先（現行 `walking_smoking.py` のコメント参照）。

### config-driven

- 現場キャリブレーションをコード変更なしで回すため。

### デモサイトのフォールバック JSON

- GitHub Pages は静的ホスティング。`fetch` 失敗やインデックス未生成でも体験が止まらないよう `app.js` に埋め込みデータあり。

---

## 実行コマンドリファレンス

```bash
cd /workspace/ai_coding_ws/urban-behavior-detector
pip install -r requirements.txt

# 動画 1 本
python scripts/run_demo.py path/to/video.mp4

# タバコモデルあり（README と同様）
python scripts/run_demo.py path/to/video.mp4 --config configs/rules_cigarette.yaml

# 出力先指定・動画スキップ
python scripts/run_demo.py video.mp4 -ov result.mp4 -oj result.json
python scripts/run_demo.py video.mp4 --no-video

# ストリーム
python scripts/run_stream.py 0 --display
python scripts/run_stream.py rtsp://192.168.1.100:554/stream

# VLM（Ollama）
python scripts/vlm_evaluate.py outputs/demo_events/some_events.json

# VLM（OpenAI 互換 / vLLM on Jetson 等）
python scripts/vlm_evaluate.py events.json --backend openai --base-url http://jetson:8000 --model Qwen2.5-VL-3B
python scripts/vlm_evaluate.py events.json --rate-limit 2 --timeout 60

# ROI キャリブレーション（信号・歩道ゾーンを GUI で描画）
python scripts/calibrate_roi.py video.mp4
python scripts/calibrate_roi.py video.mp4 --frame 100 --output my_rois.yaml

# 構造化ログ
python scripts/run_demo.py video.mp4 --log-jsonl output.jsonl
python scripts/run_stream.py 0 --display --log-jsonl stream.jsonl

# 品質
pytest tests/ -v
ruff check .

# デモアセット
python scripts/export_demo_assets.py
cd docs && python -m http.server 8000
```

---

## デモサイト構成（参照）

- `docs/index.html` セクション構成・`docs/assets/js/app.js` のタブ／フィルタ／動画連動は従来どおり。
- アセットの真実のソースは `outputs/` → `scripts/export_demo_assets.py` 経由が基本。

---

## 更新履歴（本ドキュメント）

| 日付 | 内容 |
|------|------|
| 2026-04-08 | Claude 引き継ぎ向けに全面拡充。cigarette / walking_phone / vlm_evaluate / Phase 8+ / Jetson・Cosmos メモを追加。旧 PLAN の歩きタバコ記述を現行実装に同期。 |
| 2026-04-08 | Phase 8 完了: VLM バックエンド抽象化、レート制御、ROI クロップ、キャリブレーション GUI、メモリリーク修正、構造化ログ。Jetson VLM 推奨を Qwen2.5-VL-3B に更新。Phase 9+ バックログ整理。 |
