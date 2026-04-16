# MolmoWeb Tester

MolmoWeb Tester is a small local GUI for testing [MolmoWeb](https://github.com/allenai/molmoweb) with a browser-based control panel.

This project is currently tailored for macOS, especially Apple Silicon Macs. The model server has been adjusted to work with `mps` first and fall back to CPU when needed. Windows support is possible later, but this repository should be treated as macOS-first for now.

## English

### What This Project Does

- Starts the MolmoWeb model server on your local machine
- Adds a local browser GUI for sending natural-language tasks
- Shows the current run status, current step, and execution trace
- Stores run history locally so you can reopen previous results
- Lets you reset the browser session without restarting the whole app

### Current Scope

- OS: macOS
- Recommended hardware: Apple Silicon Mac
- Model: tested with `MolmoWeb-8B`
- Browser control: Playwright-managed Chromium
- UI language: Japanese / English switcher

This is not a hosted web product. It is a local tester that opens:

- a model server at `http://127.0.0.1:8001`
- a local GUI at `http://127.0.0.1:8010`

### Repository Layout

- `agent/`
  Model server and backend selection
- `inference/client.py`
  MolmoWeb client with local browser session support
- `inference/gui_app.py`
  Local FastAPI GUI for testing MolmoWeb
- `scripts/start_server.sh`
  Starts the model server
- `scripts/start_gui.sh`
  Starts the tester GUI

### Requirements

- macOS
- Python 3.10+
- `uv`
- Playwright Chromium dependencies
- MolmoWeb checkpoint access

Install dependencies:

```bash
uv sync
uv run playwright install chromium
```

### Quick Start

1. Start the model server:

```bash
cd /path/to/molmoweb
export PREDICTOR_TYPE=hf
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B
```

2. Start the GUI in another terminal:

```bash
cd /path/to/molmoweb
bash scripts/start_gui.sh 8010
```

3. Open:

```text
http://127.0.0.1:8010/?lang=ja
```

### How To Use The GUI

- Enter a task in the textarea
- Set the maximum step count
- Set browser window width and height
- Click `Run task`
- Watch the current run and execution steps on the right
- Open `History` to review previous runs
- Click `Reset browser session` if you want a fresh Chromium session

### Notes

- Chromium is controlled in a separate window
- The tester does not control your existing Safari or Brave session
- Right-click and some advanced browser behaviors may still be imperfect
- Run history is saved locally in `inference/htmls/gui/history.json`

### Roadmap

- Better Windows support
- Better right-click and tab handling
- Stronger task progress feedback
- Optional live screenshot panel

## 日本語

### このプロジェクトについて

MolmoWeb Tester は、MolmoWeb をローカル環境で試すための簡易 GUI です。自然言語のタスクを入力すると、MolmoWeb が別ウィンドウの Chromium を操作し、その実行状況をブラウザ上から確認できます。

このリポジトリは現時点では macOS 向けです。特に Apple Silicon Mac を前提に調整しており、モデルサーバーは `mps` を優先し、使えない場合は CPU にフォールバックします。Windows 対応は今後の拡張候補です。

### できること

- ローカルで MolmoWeb のモデルサーバーを起動
- ブラウザ GUI からタスクを送信
- 現在の実行状況とステップを確認
- 実行履歴をローカルに保存
- ブラウザセッションだけを個別にリセット
- 日本語 / 英語を切り替え

### 現在の前提

- 対応 OS: macOS
- 推奨環境: Apple Silicon Mac
- 動作確認モデル: `MolmoWeb-8B`
- ブラウザ操作: Playwright 管理下の Chromium

このプロジェクトはクラウドサービスではなく、ローカルで次の 2 つを起動して使います。

- モデルサーバー: `http://127.0.0.1:8001`
- GUI: `http://127.0.0.1:8010`

### 主なファイル

- `agent/`
  モデルサーバーとバックエンド切り替え
- `inference/client.py`
  ローカルブラウザセッション付きの MolmoWeb クライアント
- `inference/gui_app.py`
  MolmoWeb Tester の GUI 本体
- `scripts/start_server.sh`
  モデルサーバー起動用スクリプト
- `scripts/start_gui.sh`
  GUI 起動用スクリプト

### 必要なもの

- macOS
- Python 3.10 以上
- `uv`
- Playwright の Chromium
- MolmoWeb のチェックポイント

セットアップ:

```bash
uv sync
uv run playwright install chromium
```

### 起動手順

1. モデルサーバーを起動:

```bash
cd /path/to/molmoweb
export PREDICTOR_TYPE=hf
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B
```

2. 別ターミナルで GUI を起動:

```bash
cd /path/to/molmoweb
bash scripts/start_gui.sh 8010
```

3. ブラウザで開く:

```text
http://127.0.0.1:8010/?lang=ja
```

### GUI の使い方

- テキスト欄にタスクを入力
- 最大ステップ数を設定
- Chromium の横幅と高さを設定
- `タスクを実行` を押す
- 右側で現在の実行状況と実行ステップを確認
- `履歴` から過去の結果を確認
- 新しい Chromium セッションにしたいときは `セッションをリセット` を押す

### 注意点

- Chromium は別ウィンドウで開きます
- 既存の Safari や Brave のセッションは操作しません
- 右クリックや新しいタブ関連の挙動はまだ改善余地があります
- 履歴は `inference/htmls/gui/history.json` にローカル保存されます

### 今後の改善候補

- Windows 対応
- 右クリックとタブ操作の改善
- 進行状況 UI の強化
- ライブスクリーンショット表示
