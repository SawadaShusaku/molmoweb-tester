# MolmoWeb Tester

macOS 向けの MolmoWeb ローカルテスターです。

## 目次

- [概要](#概要)
- [主な機能](#主な機能)
- [対応プラットフォーム](#対応プラットフォーム)
- [必要なもの](#必要なもの)
- [インストール](#インストール)
- [起動手順](#起動手順)
- [GUI の使い方](#gui-の使い方)
- [主なファイル構成](#主なファイル構成)
- [注意点](#注意点)

## 概要

MolmoWeb Tester は、MolmoWeb のモデルサーバーに対してローカル GUI を重ね、次のような確認をしやすくするためのリポジトリです。

- 自然言語タスクの送信
- 現在の実行状況の確認
- ステップごとのスクリーンショット確認
- 過去の実行履歴の再確認

現時点では特に Apple Silicon Mac を前提に調整しています。

## 主な機能

- FastAPI ベースのローカル GUI
- 現在の実行状況と実行ステップのライブ表示
- ローカル履歴の保存
- Chromium セッションの個別リセット
- 日本語 / 英語切り替え
- Chromium の横幅・高さの指定
- 非 CUDA 環境向けの `mps` / CPU フォールバック

## 対応プラットフォーム

現在の対象:

- macOS
- Apple Silicon 推奨

今後の候補:

- Windows 対応

## 必要なもの

- Python 3.10 以上
- `uv`
- Playwright の Chromium
- MolmoWeb のチェックポイント

## インストール

```bash
uv sync
uv run playwright install chromium
```

## 起動手順

まずモデルサーバーを起動します。

```bash
cd /path/to/molmoweb
export PREDICTOR_TYPE=hf
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B
```

次に別ターミナルで GUI を起動します。

```bash
cd /path/to/molmoweb
bash scripts/start_gui.sh 8010
```

ブラウザで次を開きます。

```text
http://127.0.0.1:8010/?lang=ja
```

## GUI の使い方

1. メインのテキスト欄にタスクを入力します。
2. 最大ステップ数を設定します。
3. Chromium の横幅と高さを設定します。
4. `タスクを実行` を押します。
5. 右側で現在の実行状況と実行ステップを確認します。
6. `履歴` から過去の実行結果を確認します。
7. 新しい Chromium セッションにしたい場合は `セッションをリセット` を使います。

## 主なファイル構成

- `agent/`
  モデルサーバーと予測バックエンド
- `inference/client.py`
  MolmoWeb クライアントとブラウザセッション管理
- `inference/gui_app.py`
  ローカルテスター GUI
- `scripts/start_server.sh`
  モデルサーバー起動スクリプト
- `scripts/start_gui.sh`
  GUI 起動スクリプト
- `assets/`
  ロゴや README 用画像

## 注意点

- Chromium は別ウィンドウで動作します。
- 既存の Safari や Brave のセッションを直接操作するものではありません。
- 右クリックやタブ操作など、一部挙動はまだ改善余地があります。
- 履歴ファイルは `inference/htmls/gui/` 配下に保存されます。
