# Urban Behavior Detector - 実装計画

## プロジェクト概要

ライブカメラ映像や動画から都市における迷惑行為・違反行動を検知するOSS。
GitHub Pages で「コードを見る前にデモを触って理解できる」プロダクトとして設計する。

---

## 現在の完了状況

### Phase 1: コア実装 [完了]

- [x] ディレクトリ構成の作成
- [x] `detectors/yolo_detector.py` - YOLOv8ラッパー（person/bicycle/phone/umbrella/bottle等）
- [x] `trackers/sort_tracker.py` - IoUベースSORT追跡（速度・方向算出付き）
- [x] `behaviors/base.py` - BehaviorAnalyzer基底クラス + ViolationEventデータクラス
- [x] `behaviors/walking_smoking.py` - 歩きタバコ検知（手-口近接 + 移動速度 + 持続時間）
- [x] `behaviors/bicycle_violation.py` - 自転車スマホ運転 + 傘差し運転検知
- [x] `pipelines/video_pipeline.py` - Detection→Tracking→Behavior→Output統合パイプライン
- [x] `scripts/run_demo.py` - CLIエントリポイント
- [x] `scripts/export_demo_assets.py` - outputs/ → docs/assets/ エクスポート
- [x] `configs/rules.yaml` - 全閾値・有効化フラグ（違反タイプ独立）
- [x] `requirements.txt` - ultralytics, opencv-python, numpy, pyyaml

### Phase 2: デモサイト [完了]

- [x] `docs/index.html` - 6セクション構成のシングルページ
- [x] `docs/assets/css/style.css` - ダークテーマ、モバイル対応、違反タイプ別カラー
- [x] `docs/assets/js/app.js` - タブ切替、イベントログ、タイムラインUI、JSON構文ハイライト
- [x] サンプルJSON（sample_crosswalk, sample_station の2動画分、計7イベント）
- [x] `README.md`

---

### Phase 3: 実動画での検証 [完了]

- [x] Mixkit CCライセンス動画3本を入手（東京歩行者, 路面電車＋歩行者, 都市自転車）
- [x] `python scripts/run_demo.py` を実動画で実行・正常動作確認
- [x] アノテーション付き出力動画の生成確認
- [x] 出力動画をdocs/assets/videosに配置（ffmpegでH.264再エンコード、73MB→13MB）
- [x] デモアセットエクスポート（サンプル2 + 実検知3 = 計5エントリ）

### Phase 4: 精度改善 [完了]

- [x] 歩きタバコ: YOLOv8-pose で手首-鼻キーポイント距離ベースの検知を実装（proxy方式もフォールバックとして維持）
- [x] 自転車スマホ: person-bicycle対応付けをIoU+中心距離+速度類似性のハイブリッドスコアリングに改善
- [x] confidence計算: 線形増加→持続時間+一貫性（連続フレーム比率）ベースに改善（`compute_confidence()` 共通関数化）

### Phase 5: 追加違反タイプ [完了]

- [x] **逆走検知**: `behaviors/wrong_way.py` - 方向角度 + 許容差で判定
- [x] **信号無視検知**: `behaviors/signal_violation.py` + `detectors/signal_detector.py` - YOLO traffic light検出 + HSV色分類
- [x] **歩道走行検知**: `behaviors/sidewalk_riding.py` - ROIポリゴンベース（セグメンテーション不要の軽量実装）

### Phase 6: デモサイト強化 [完了]

- [x] 動画プレーヤー（video要素での再生、アノテーション付き動画対応）
- [x] 動画再生とイベントログの連動（イベントクリック→該当時刻にシーク）
- [x] 検知結果のフィルタリングUI（6違反タイプ対応）
- [x] 統計サマリー表示（違反タイプ別カウント）
- [x] OGP / favicon の設定
- [x] 地理参照ヒートマップ（Leaflet.js + サンプルデータ）

### Phase 7: プロダクト品質 [完了]

- [x] ユニットテスト（60テスト、全モジュールカバー）
- [x] GitHub Actions CI（ruff lint + pytest）
- [x] LICENSE ファイル追加（MIT）
- [x] .gitignore 整備（__pycache__, outputs/, *.pt, samples/ 等）
- [x] Docker対応（Dockerfile + .dockerignore）
- [x] ストリーミング入力対応（RTSP, ウェブカメラ, HTTP）

---

## アーキテクチャ詳細

### パイプライン処理フロー

```
入力動画
  │
  ▼
YOLODetector.detect(frame)
  │  Detection[] (bbox, class_id, class_name, confidence)
  ▼
SORTTracker.update(detections)
  │  Track[] (track_id, bbox, center, speed, direction, history)
  ▼
BehaviorAnalyzer[].update(frame_idx, tracks, all_detections)
  │  ViolationEvent[] (type, track_id, start_frame, end_frame, confidence)
  ▼
出力: JSON + アノテーション動画
```

### 違反判定ロジック詳細

#### 歩きタバコ (walking_smoking)
```
条件:
  1. track.class_name == "person"
  2. track.speed > speed_threshold (移動中)
  3. 小物体(bottle/handbag)がpersonの口元領域に近接
     - 口元領域: body上端15-30%, 左右20%マージン
     - 距離 < person_height * hand_mouth_distance
  4. 上記が min_duration_frames 以上連続

改善案:
  - YOLOv8-pose でキーポイント取得 → 手首-鼻の距離を直接計算
  - cigarette検出の専用モデル（fine-tune）
```

#### 自転車スマホ (bicycle_phone)
```
条件:
  1. person track と bicycle track が重なっている（cyclist判定）
  2. "cell phone" 検出が cyclist の顔領域に近接
     - 顔領域: person bbox 上端35%
  3. min_duration_frames 以上連続

改善案:
  - person-bicycle のペアリング精度向上
  - phone の向き（画面が顔に向いているか）
```

#### 傘差し運転 (bicycle_umbrella)
```
条件:
  1. cyclist判定（上記と同じ）
  2. "umbrella" 検出が cyclist の bbox と overlap_threshold 以上重なる
  3. min_duration_frames 以上連続
```

### ファイル構成と責務

```
urban-behavior-detector/
├── detectors/
│   ├── __init__.py
│   ├── yolo_detector.py          # YOLOv8ラッパー、Detection dataclass
│   ├── pose_detector.py          # YOLOv8-pose ラッパー、PoseDetection
│   └── signal_detector.py        # 信号機HSV色分類、SignalDetection
├── trackers/
│   ├── __init__.py
│   └── sort_tracker.py           # SORT追跡、Track dataclass（speed/direction算出）
├── behaviors/
│   ├── __init__.py
│   ├── base.py                   # BehaviorAnalyzer ABC、ViolationEvent、compute_confidence
│   ├── walking_smoking.py        # 歩きタバコ判定（proxy + pose対応）
│   ├── bicycle_violation.py      # 自転車スマホ + 傘差し判定
│   ├── wrong_way.py              # 逆走検知
│   ├── signal_violation.py       # 信号無視検知
│   └── sidewalk_riding.py        # 歩道走行検知
├── pipelines/
│   ├── __init__.py
│   └── video_pipeline.py         # 統合パイプライン（動画/ストリーム/描画）
├── configs/
│   └── rules.yaml                # 全設定（閾値、有効化フラグ、モデル、位置情報）
├── scripts/
│   ├── run_demo.py               # CLI: 動画 → 検知結果
│   ├── run_stream.py             # CLI: ストリーム → リアルタイム検知
│   ├── export_demo_assets.py     # outputs/ → docs/assets/ コピー
│   └── generate_heatmap.py       # 複数結果 → ヒートマップデータ集約
├── tests/                        # pytest ユニットテスト（60件）
├── docs/
│   ├── index.html                # GitHub Pages（8セクション）
│   └── assets/
│       ├── css/style.css         # ダークテーマCSS
│       ├── js/app.js             # デモUI（タブ、フィルタ、統計、動画プレーヤー）
│       ├── js/heatmap.js         # Leaflet.js ヒートマップ
│       ├── videos/               # アノテーション付き検知動画
│       └── json/                 # サンプル + 検知JSON + ヒートマップデータ
├── Dockerfile                    # Python 3.12-slim + OpenCV
├── .dockerignore
├── .github/workflows/ci.yml     # ruff lint + pytest CI
├── README.md
├── LICENSE                       # MIT
├── PLAN.md                       # この文書
└── requirements.txt
```

### configs/rules.yaml の構造

```yaml
# 各違反タイプ: enabled + 固有閾値
walking_smoking:
  enabled: bool
  hand_mouth_distance: float    # body height比
  speed_threshold: float        # px/frame
  min_duration_frames: int
  confidence_threshold: float

bicycle_phone:
  enabled: bool
  phone_near_face_threshold: float
  min_duration_frames: int
  confidence_threshold: float

bicycle_umbrella:
  enabled: bool
  umbrella_overlap_threshold: float
  min_duration_frames: int
  confidence_threshold: float

bicycle_wrong_way:
  enabled: bool                 # デフォルトfalse
  expected_direction: str
  angle_tolerance: int
  min_duration_frames: int
  confidence_threshold: float

# グローバル設定
detection:
  model: str                    # YOLOモデルパス
  confidence: float
  iou_threshold: float
  classes: list[int]            # COCO class IDs

tracking:
  max_age: int
  min_hits: int
  iou_threshold: float
```

### デモサイト構成

```
index.html セクション:
  1. Hero        - プロジェクト名、一行説明、GitHub link、Demo導線
  2. Violations  - 3カード（歩きタバコ、自転車スマホ、傘差し）
  3. How it works - 4ステップパイプラインフロー
  4. Demo        - タブ切替、動画プレースホルダー、タイムライン、イベントログ
  5. JSON Preview - 構文ハイライト付きJSON表示
  6. Roadmap     - 6項目（Shipped / Next / Planned）

JS機能:
  - demo_index.json を fetch → タブ生成
  - タブ切替でイベントログ・タイムライン・JSONプレビュー更新
  - fetch失敗時は埋め込みフォールバックデータ使用
  - タイムラインは違反タイプ別色分け（赤/橙/黄）
  - JSONは key/string/number の構文ハイライト
```

---

## 設計判断の記録

### なぜSORT (not ByteTrack/DeepSORT)?
- MVPとして最小限のIoUベース追跡で十分
- 外部依存なし（numpy のみ）
- ByteTrack/DeepSORTへの差し替えはTrack I/Fが同じなら容易

### なぜプロキシ物体で歩きタバコ判定?
- COCO事前学習YOLOに cigarette クラスがない
- bottle/handbag を代用（小物体 + 口元近接 + 移動中 の条件で絞り込み）
- 改善パスとして pose estimation への切替を明記

### なぜ config-driven?
- 同じパイプラインで異なるシーン・カメラに対応
- 違反タイプの有効/無効を切替可能
- 閾値チューニングにコード変更不要

### デモサイトになぜフォールバックデータ?
- GitHub Pages では静的ファイルのみ
- demo_index.json が未生成でもデモが動く
- 初見ユーザーがすぐ体験できる

---

## 実行コマンドリファレンス

```bash
# セットアップ
pip install -r requirements.txt

# 基本実行
python scripts/run_demo.py path/to/video.mp4

# カスタム出力先
python scripts/run_demo.py video.mp4 -ov result.mp4 -oj result.json

# JSON出力のみ（動画生成スキップ）
python scripts/run_demo.py video.mp4 --no-video

# カスタム設定
python scripts/run_demo.py video.mp4 --config my_rules.yaml

# デモアセットエクスポート
python scripts/export_demo_assets.py

# ローカルでデモサイト確認
cd docs && python -m http.server 8000
```
