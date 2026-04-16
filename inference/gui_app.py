"""
Minimal local web UI for driving MolmoWeb from a browser.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent.actions import SendMsgToUser
from inference import MolmoWeb, Trajectory


ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT_DIR / "inference" / "htmls" / "gui"
HTML_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = HTML_DIR / "history.json"

DEFAULT_ENDPOINT = os.environ.get("MOLMOWEB_GUI_ENDPOINT", "http://127.0.0.1:8001")
DEFAULT_MAX_STEPS = int(os.environ.get("MOLMOWEB_GUI_MAX_STEPS", "10"))
HEADLESS = os.environ.get("MOLMOWEB_GUI_HEADLESS", "false").lower() == "true"
LANGS = {"en", "ja"}

TEXT = {
    "en": {
        "title": "MolmoWeb Tester",
        "subtitle": "unused",
        "help_heading": "How to use",
        "help_steps": [
            ("1.", "Enter a task", "Type what you want the browser to do."),
            ("2.", "Adjust settings", "Set the step limit and window size if needed."),
            ("3.", "Run", 'Press "Run task" — the right pane shows live progress.'),
            ("4.", "Review", "Check past results from the History panel."),
        ],
        "help_note": "Keep the model server running at",
        "help_button": "?",
        "browser_session": "Browser Session",
        "headless": "Headless",
        "ready": "Ready",
        "not_started": "Not started yet",
        "task": "Task",
        "task_placeholder": "Go to arxiv.org and find the paper about Molmo and Pixmo.",
        "max_steps": "Max steps",
        "window_size": "Window size",
        "width": "Width",
        "height": "Height",
        "window_size_invalid": "Enter width and height as numbers, at least 640 and 480.",
        "run_task": "Run task",
        "running": "Running",
        "reset": "Reset browser session",
        "force_stop": "Force stop",
        "history": "Run History",
        "history_toggle": "History",
        "history_close": "Close",
        "delete": "Delete",
        "clear_history": "Clear all",
        "no_runs": "No runs yet.",
        "task_finished": "Task finished.",
        "task_failed": "Task failed before completion.",
        "enter_task": "Please enter a task before sending it.",
        "prompt": "Prompt",
        "answer": "Answer",
        "steps": "steps",
        "trajectory": "trajectory",
        "error": "error",
        "completed": "completed",
        "answered": "answered",
        "incomplete": "incomplete",
        "stopped": "stopped",
        "current_run": "Current Run",
        "current_page": "Current page",
        "page_title": "Page title",
        "live_trace": "Execution Steps",
        "expand_all": "Expand all",
        "idle": "Idle",
        "current_action": "Current action",
        "current_url": "Current URL",
        "current_step": "Current step",
        "queued": "Queued",
        "lang_ja": "日本語",
        "lang_en": "English",
    },
    "ja": {
        "title": "MolmoWeb Tester",
        "subtitle": "unused",
        "help_heading": "使い方",
        "help_steps": [
            ("1.", "タスクを入力", "ブラウザにやらせたい操作を書きます。"),
            (
                "2.",
                "設定を調整",
                "必要に応じてステップ数やウィンドウサイズを変更します。",
            ),
            (
                "3.",
                "実行",
                "「タスクを実行」を押すと、右側にリアルタイムで進捗が表示されます。",
            ),
            ("4.", "結果を確認", "過去の実行結果は「履歴」パネルから確認できます。"),
        ],
        "help_note": "モデルサーバーを起動したままにしてください :",
        "help_button": "？",
        "browser_session": "ブラウザセッション",
        "headless": "ヘッドレス",
        "ready": "準備完了",
        "not_started": "未起動",
        "task": "タスク",
        "task_placeholder": "arxiv.org に行って、Molmo と Pixmo の論文を見つけてください。",
        "max_steps": "最大ステップ数",
        "window_size": "ウィンドウサイズ",
        "width": "横幅",
        "height": "高さ",
        "window_size_invalid": "横幅と高さは 640 と 480 以上の数値で入力してください。",
        "run_task": "タスクを実行",
        "running": "実行中",
        "reset": "セッションをリセット",
        "force_stop": "強制停止",
        "history": "実行履歴",
        "history_toggle": "履歴",
        "history_close": "閉じる",
        "delete": "削除",
        "clear_history": "すべて削除",
        "no_runs": "まだ実行履歴はありません。",
        "task_finished": "タスクが完了しました。",
        "task_failed": "タスクは完了前に失敗しました。",
        "enter_task": "送信前にタスクを入力してください。",
        "prompt": "指示",
        "answer": "回答",
        "steps": "ステップ数",
        "trajectory": "軌跡",
        "error": "エラー",
        "completed": "完了",
        "answered": "回答済み",
        "incomplete": "未完了",
        "stopped": "停止",
        "current_run": "現在の実行",
        "current_page": "現在のページ",
        "page_title": "ページタイトル",
        "live_trace": "実行ステップ",
        "expand_all": "すべて展開",
        "idle": "待機中",
        "current_action": "現在のアクション",
        "current_url": "現在の URL",
        "current_step": "現在のステップ",
        "queued": "キュー済み",
        "lang_ja": "日本語",
        "lang_en": "English",
    },
}


@dataclass
class RunRecord:
    id: str
    prompt: str
    answer: str
    status: str
    max_steps: int
    created_at: str
    trajectory_href: str | None = None
    final_url: str | None = None
    final_title: str | None = None
    step_count: int = 0
    error: str | None = None


@dataclass
class LiveRun:
    running: bool = False
    prompt: str = ""
    max_steps: int = 0
    step_num: int = 0
    action: str = ""
    page_url: str = ""
    page_title: str = ""
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""
    answer: str = ""
    status: str = "idle"
    trajectory_href: str | None = None
    steps_log: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AppState:
    endpoint: str = DEFAULT_ENDPOINT
    max_steps: int = DEFAULT_MAX_STEPS
    headless: bool = HEADLESS
    local: bool = True
    viewport_width: int = 1600
    viewport_height: int = 1000
    client: MolmoWeb | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    history: list[RunRecord] = field(default_factory=list)
    last_error: str | None = None
    live_run: LiveRun = field(default_factory=LiveRun)
    worker: threading.Thread | None = None
    task_queue: queue.Queue[tuple[str, int] | None] = field(default_factory=queue.Queue)
    stop_requested: bool = False

    def get_client(self) -> MolmoWeb:
        if self.client is None:
            self.client = MolmoWeb(
                endpoint=self.endpoint,
                local=self.local,
                keep_alive=True,
                headless=self.headless,
                verbose=True,
                step_callback=self.update_live_run,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
            )
        else:
            self.client.step_callback = self.update_live_run
            self.client.viewport_width = self.viewport_width
            self.client.viewport_height = self.viewport_height
        return self.client

    def reset_browser(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.last_error = None
        self.stop_requested = False
        self.live_run = LiveRun()

    def force_stop(self) -> None:
        self.stop_requested = True
        if self.client is not None:
            self.client.close()
            self.client = None
        self.last_error = None
        self.live_run.running = False
        self.live_run.status = "stopped"
        self.live_run.error = None
        self.live_run.answer = "Force stopped"
        self.live_run.finished_at = datetime.now().strftime("%H:%M:%S")

    def update_live_run(self, event: dict) -> None:
        with self.lock:
            self.live_run.step_num = event.get("step_num", self.live_run.step_num)
            self.live_run.max_steps = event.get("max_steps", self.live_run.max_steps)
            self.live_run.action = event.get("action") or ""
            self.live_run.page_url = event.get("page_url") or ""
            self.live_run.page_title = event.get("page_title") or ""
            self.live_run.error = event.get("error")
            step_num = event.get("step_num")
            action = event.get("action") or ""
            if step_num and action:
                item = {
                    "summary": f"{step_num}. {action}",
                    "screenshot_base64": event.get("screenshot_base64") or "",
                    "page_url": event.get("page_url") or "",
                    "page_title": event.get("page_title") or "",
                }
                if (
                    not self.live_run.steps_log
                    or self.live_run.steps_log[-1]["summary"] != item["summary"]
                ):
                    self.live_run.steps_log.append(item)
            if self.live_run.running:
                self.live_run.status = "running"

    def ensure_worker(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=_worker_loop, daemon=True)
            self.worker.start()


def _load_history() -> list[RunRecord]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text())
        if not isinstance(data, list):
            return []
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item.setdefault("id", uuid4().hex)
            records.append(RunRecord(**item))
        return records
    except Exception:
        return []


def _save_history() -> None:
    HISTORY_PATH.write_text(
        json.dumps(
            [asdict(record) for record in STATE.history], ensure_ascii=False, indent=2
        )
    )


STATE = AppState()
STATE.history = _load_history()
app = FastAPI(title="MolmoWeb GUI")
app.mount(
    "/artifacts",
    StaticFiles(directory=str(ROOT_DIR / "inference" / "htmls")),
    name="artifacts",
)
app.mount("/assets", StaticFiles(directory=str(ROOT_DIR / "assets")), name="assets")


def _lang_value(lang: str | None) -> str:
    return lang if lang in LANGS else "en"


def _t(lang: str, key: str) -> str:
    return TEXT[_lang_value(lang)][key]


def _extract_answer(traj: Trajectory) -> tuple[str, str]:
    for step in reversed(traj.steps):
        if step.prediction and isinstance(step.prediction.action, SendMsgToUser):
            msg = step.prediction.action.msg.strip()
            if msg.startswith("[ANSWER]"):
                return msg.replace("[ANSWER]", "", 1).strip() or "Done", "answered"
            if msg.startswith("[EXIT]"):
                return "Done", "completed"
        if step.error:
            return step.error, "error"
    return "No final answer was produced.", "incomplete"


def _save_trajectory(traj: Trajectory, prompt: str) -> str:
    slug = (
        "".join(ch if ch.isalnum() else "_" for ch in prompt).strip("_")[:40] or "task"
    )
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}_{uuid4().hex[:8]}.html"
    output_path = HTML_DIR / filename
    traj.save_html(output_path=str(output_path), query=prompt)
    return f"/artifacts/gui/{filename}"


def _truncate_text(text: str | None, max_chars: int = 140) -> str:
    if not text:
        return "-"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_record(record: RunRecord, lang: str) -> str:
    status = (
        _t(lang, record.status)
        if record.status in {"completed", "answered", "incomplete", "stopped"}
        else record.status
    )
    detail_bits = [
        f"{escape(_t(lang, 'steps'))}: {record.step_count}",
        f"max_steps: {record.max_steps}",
    ]
    if record.final_title:
        detail_bits.append(f"title: {escape(record.final_title)}")
    if record.final_url:
        detail_bits.append(f"url: {escape(record.final_url)}")
    if record.trajectory_href:
        detail_bits.append(
            f'<a href="{escape(record.trajectory_href)}" target="_blank" rel="noreferrer">{escape(_t(lang, "trajectory"))}</a>'
        )
    error_html = (
        f'<p class="meta-line">{escape(_t(lang, "error"))}: {escape(record.error)}</p>'
        if record.error
        else ""
    )
    prompt_preview = escape(_truncate_text(record.prompt, 110))
    answer_preview = escape(_truncate_text(record.answer, 150))
    full_prompt = escape(record.prompt)
    full_answer = escape(record.answer)
    return f"""
    <article class="record" onclick="this.classList.toggle('expanded')">
      <div class="record-header">
        <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        <div class="record-meta">
          <span>{escape(record.created_at)}</span>
          <span>{escape(status)}</span>
        </div>
        <button class="delete-btn" type="button" data-record-id="{escape(record.id)}" title="{escape(_t(lang, "delete"))}">&#x1F5D1;&#xFE0E;</button>
      </div>
      <p class="record-copy"><strong>{escape(_t(lang, "prompt"))}</strong><br><span class="preview">{prompt_preview}</span><span class="full" hidden>{full_prompt}</span></p>
      <p class="answer"><strong>{escape(_t(lang, "answer"))}</strong><br><span class="preview">{answer_preview}</span><span class="full" hidden>{full_answer}</span></p>
      <p class="meta-line">{" | ".join(detail_bits)}</p>
      {error_html}
    </article>
    """


def _status_payload(lang: str) -> dict:
    live = asdict(STATE.live_run)
    live["browser_status"] = (
        _t(lang, "ready") if STATE.client is not None else _t(lang, "not_started")
    )
    live["localized_status"] = (
        _t(lang, live["status"])
        if live["status"] in {"completed", "answered", "incomplete", "stopped"}
        else (_t(lang, "running") if live["running"] else _t(lang, "idle"))
    )
    live["history_html"] = "".join(
        _render_record(record, lang) for record in reversed(STATE.history)
    ) or (f'<p class="hint">{escape(_t(lang, "no_runs"))}</p>')
    live["trace_html"] = (
        "".join(
            _render_trace_item(item, idx) for idx, item in enumerate(live["steps_log"])
        )
        or f'<p class="hint">{escape(_t(lang, "idle"))}</p>'
    )
    live["last_error"] = STATE.last_error
    # Latest step content for the "current step" panel
    steps = live["steps_log"]
    if len(steps) >= 2:
        latest = steps[-1]
        live["latest_step_html"] = _render_latest_step(latest)
    elif len(steps) == 1:
        live["latest_step_html"] = f'<p class="hint">{escape(_t(lang, "idle"))}</p>'
    else:
        live["latest_step_html"] = f'<p class="hint">-</p>'
    return live


def _render_latest_step(item: dict[str, str]) -> str:
    parts = []
    summary = item.get("summary", "")
    if summary:
        parts.append(f'<div class="latest-action">{escape(summary)}</div>')
    meta = []
    if item.get("page_title"):
        meta.append(f"title: {escape(item['page_title'])}")
    if item.get("page_url"):
        meta.append(f"url: {escape(item['page_url'])}")
    if meta:
        parts.append(f'<div class="trace-meta">{" | ".join(meta)}</div>')
    if item.get("screenshot_base64"):
        parts.append(
            f'<img class="trace-shot" alt="step screenshot" '
            f'src="data:image/png;base64,{item["screenshot_base64"]}">'
        )
    return "".join(parts)


def _render_trace_item(item: dict[str, str], idx: int = 0) -> str:
    is_first = idx == 0
    img_html = ""
    if not is_first and item.get("screenshot_base64"):
        img_html = (
            f'<img class="trace-shot" alt="step screenshot" '
            f'src="data:image/png;base64,{item["screenshot_base64"]}">'
        )
    meta = []
    if item.get("page_title"):
        meta.append(f"title: {escape(item['page_title'])}")
    if not is_first and item.get("page_url"):
        meta.append(f"url: {escape(item['page_url'])}")
    meta_html = f'<div class="trace-meta">{" | ".join(meta)}</div>' if meta else ""
    open_attr = "" if is_first else " open"
    return (
        f'<details class="trace-item"{open_attr}>'
        f"<summary>{escape(item.get('summary', ''))}</summary>"
        f"{meta_html}"
        f"{img_html}"
        "</details>"
    )


def _finalize_run(prompt: str, max_steps: int, traj: Trajectory) -> None:
    answer, status = _extract_answer(traj)
    if STATE.stop_requested:
        answer, status = "Force stopped", "stopped"
    href = _save_trajectory(traj, prompt)
    last_state = traj.steps[-1].state if traj.steps else None
    error = next((step.error for step in reversed(traj.steps) if step.error), None)
    record = RunRecord(
        id=uuid4().hex,
        prompt=prompt,
        answer=answer,
        status=status,
        max_steps=max_steps,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trajectory_href=href,
        final_url=last_state.page_url if last_state else None,
        final_title=last_state.page_title if last_state else None,
        step_count=len(traj.steps),
        error=error,
    )
    with STATE.lock:
        STATE.history.append(record)
        _save_history()
        STATE.last_error = error
        STATE.live_run.running = False
        STATE.live_run.answer = answer
        STATE.live_run.error = error
        STATE.live_run.status = status
        STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")
        STATE.live_run.trajectory_href = href
        STATE.stop_requested = False


def _worker_loop() -> None:
    while True:
        item = STATE.task_queue.get()
        if item is None:
            break
        prompt, max_steps = item
        with STATE.lock:
            STATE.last_error = None
        client = STATE.get_client()
        try:
            traj = client.run(query=prompt, max_steps=max_steps)
            _finalize_run(prompt, max_steps, traj)
        except Exception as exc:
            with STATE.lock:
                if STATE.stop_requested:
                    STATE.last_error = None
                    STATE.live_run.running = False
                    STATE.live_run.error = None
                    STATE.live_run.status = "stopped"
                    STATE.live_run.answer = "Force stopped"
                    STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")
                    STATE.stop_requested = False
                    continue
                STATE.last_error = str(exc)
                STATE.live_run.running = False
                STATE.live_run.error = str(exc)
                STATE.live_run.status = "error"
                STATE.live_run.finished_at = datetime.now().strftime("%H:%M:%S")


def _render_page(lang: str = "en") -> HTMLResponse:
    lang = _lang_value(lang)
    data = _status_payload(lang)
    current_step = (
        escape(_t(lang, "queued"))
        if data["running"] and not data["step_num"]
        else f"{data['step_num']}/{data['max_steps']}"
        if data["step_num"]
        else "-"
    )

    html = f"""<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MolmoWeb GUI</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --panel: #fffdf9;
      --ink: #1e2430;
      --muted: #5f6b7a;
      --line: #d7cfbf;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --error: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      color: var(--ink);
      font-size: 18px;
      background: linear-gradient(180deg, #faf7f0 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1600px, calc(100vw - 32px));
      margin: 0 auto;
      padding: clamp(16px, 3vw, 32px) 0 56px;
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .brand {{
      display: inline-flex;
      align-items: center;
      gap: 14px;
    }}
    .brand-mark {{
      width: 42px;
      height: 42px;
      object-fit: contain;
      flex: 0 0 auto;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.8rem);
      line-height: 0.95;
      color: #f0529c;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: clamp(1.02rem, 1.45vw, 1.14rem);
      max-width: 72ch;
    }}
    .top-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .help-wrap {{
      position: relative;
    }}
    .help-button {{
      width: 46px;
      height: 46px;
      justify-content: center;
      padding: 0;
      font-size: 1.08rem;
    }}
    .help-popover {{
      position: absolute;
      top: calc(100% + 10px);
      right: 0;
      width: min(460px, calc(100vw - 32px));
      padding: 20px 22px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.97);
      box-shadow: 0 18px 40px rgba(30, 36, 48, 0.13);
      color: var(--ink);
      font: 500 0.92rem/1.6 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      z-index: 30;
      opacity: 0;
      transform: translateY(-6px);
      pointer-events: none;
      transition: opacity 0.2s ease, transform 0.2s ease;
    }}
    .help-popover.open {{
      opacity: 1;
      transform: translateY(0);
      pointer-events: auto;
    }}
    .help-heading {{
      margin: 0 0 12px;
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .help-steps {{
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }}
    .help-step {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 2px;
    }}
    .help-step dt {{
      font-weight: 600;
      font-size: 0.92rem;
    }}
    .help-step .step-num {{
      color: var(--accent);
      font-weight: 700;
      margin-right: 2px;
    }}
    .help-step dd {{
      margin: 0;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.5;
      padding-left: 1.2em;
    }}
    .help-note {{
      margin: 14px 0 0;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      font-size: 0.84rem;
      color: var(--muted);
    }}
    .help-note code {{
      font-size: 0.82rem;
      background: rgba(0,0,0,0.04);
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .lang-switch {{
      display: inline-flex;
      gap: 8px;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      backdrop-filter: blur(10px);
    }}
    .lang-switch a {{
      text-decoration: none;
      padding: 10px 16px;
      border-radius: 999px;
      font: 700 0.94rem/1 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      color: var(--muted);
    }}
    .lang-switch a.active {{
      background: var(--ink);
      color: white;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(420px, 1.06fr) minmax(360px, 0.94fr);
      gap: clamp(14px, 2vw, 22px);
      align-items: start;
    }}
    .panel {{
      width: 100%;
      min-width: 0;
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border: 1px solid color-mix(in srgb, var(--line) 75%, #ffffff);
      border-radius: 20px;
      padding: clamp(18px, 2.4vw, 26px);
      box-shadow: 0 16px 44px rgba(30, 36, 48, 0.08);
    }}
    .side-stack {{
      display: grid;
      gap: 18px;
      position: sticky;
      top: 18px;
      align-self: start;
    }}
    form {{ width: 100%; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
      font: 600 1.08rem/1.5 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
    }}
    .meta > div {{
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(30, 36, 48, 0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,244,236,0.92));
    }}
    .meta strong {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    label {{
      display: block;
      margin-bottom: 10px;
      font: 700 1.06rem/1.3 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
    }}
    textarea, input[type="number"] {{
      width: 100%;
      border: 1px solid color-mix(in srgb, var(--line) 82%, #ffffff);
      border-radius: 16px;
      padding: 16px 18px;
      font: 500 1.1rem/1.6 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      background: #fffefb;
      color: var(--ink);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }}
    input[type="text"] {{
      width: 100%;
      border: 1px solid color-mix(in srgb, var(--line) 82%, #ffffff);
      border-radius: 16px;
      padding: 16px 18px;
      font: 500 1.1rem/1.6 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      background: #fffefb;
      color: var(--ink);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }}
    textarea:focus, input[type="number"]:focus, input[type="text"]:focus {{
      outline: none;
      border-color: color-mix(in srgb, var(--accent) 45%, white);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: minmax(180px, 0.9fr) minmax(140px, 0.7fr) minmax(140px, 0.7fr);
      gap: 12px;
      align-items: end;
    }}
    .field-grid-head {{
      margin-top: 14px;
      margin-bottom: 10px;
      color: var(--muted);
      font: 700 0.92rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .field-group {{
      display: grid;
      gap: 10px;
    }}
    textarea {{
      min-height: clamp(220px, 30vh, 360px);
      resize: vertical;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 14px 20px;
      font: 700 1.04rem/1 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }}
    .primary {{
      background: var(--ink);
      color: white;
    }}
    .secondary {{
      background: transparent;
      color: var(--accent-2);
      border: 1px solid color-mix(in srgb, var(--accent-2) 35%, white);
    }}
    .primary[disabled], .secondary[disabled] {{
      opacity: 0.7;
      cursor: progress;
    }}
    .spinner {{
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 2px solid rgba(255,255,255,0.35);
      border-top-color: white;
      animation: spin 0.8s linear infinite;
      display: none;
    }}
    .is-running .spinner {{ display: inline-block; }}
    .hint, .notice, .error {{
      margin: 12px 0 0;
      font: 500 1rem/1.6 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
    }}
    .hint {{ color: var(--muted); }}
    .notice {{ color: var(--accent); }}
    .error {{ color: var(--error); }}
    .live-run {{
      border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--line));
      background: linear-gradient(180deg, rgba(15,118,110,0.06), rgba(255,255,255,0.72));
    }}
    .live-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
      font: 700 1rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
    }}
    .toggle-switch {{
      position: relative;
      display: inline-block;
      width: 36px;
      height: 20px;
      cursor: pointer;
      flex-shrink: 0;
    }}
    .toggle-switch input {{
      opacity: 0;
      width: 0;
      height: 0;
    }}
    .toggle-slider {{
      position: absolute;
      inset: 0;
      background: #ccc;
      border-radius: 20px;
      transition: background 0.2s;
    }}
    .toggle-slider::before {{
      content: "";
      position: absolute;
      width: 16px;
      height: 16px;
      left: 2px;
      bottom: 2px;
      background: #fff;
      border-radius: 50%;
      transition: transform 0.2s;
    }}
    .toggle-switch input:checked + .toggle-slider {{
      background: var(--accent);
    }}
    .toggle-switch input:checked + .toggle-slider::before {{
      transform: translateX(16px);
    }}
    .toggle-switch-inline {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
    }}
    .toggle-switch-inline input {{
      opacity: 0;
      width: 0;
      height: 0;
      position: absolute;
    }}
    .toggle-switch-inline .toggle-slider {{
      position: relative;
      display: flex;
      align-items: center;
      width: 36px;
      height: 20px;
      background: #ccc;
      border-radius: 20px;
      transition: background 0.2s;
      flex-shrink: 0;
    }}
    .toggle-switch-inline .toggle-slider::before {{
      content: "";
      position: absolute;
      width: 16px;
      height: 16px;
      left: 2px;
      bottom: 2px;
      background: #fff;
      border-radius: 50%;
      transition: transform 0.2s;
    }}
    .toggle-switch-inline input:checked + .toggle-slider {{
      background: var(--accent);
    }}
    .toggle-switch-inline input:checked + .toggle-slider::before {{
      transform: translateX(16px);
    }}
    .toggle-label {{
      font: 500 0.82rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      color: var(--muted);
      line-height: 20px;
    }}
    .live-head-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .live-run-card {{
      border-radius: 18px;
      overflow: hidden;
    }}
    .progress-bar {{
      display: flex;
      gap: 4px;
      padding: 0 16px 8px;
      height: 16px;
    }}
    .progress-bar:empty {{
      display: none;
    }}
    .progress-seg {{
      flex: 1;
      border-radius: 5px;
      background: rgba(0,0,0,0.07);
      transition: background 0.4s ease;
      position: relative;
      overflow: hidden;
    }}
    .progress-seg.filled {{
      background: var(--accent);
    }}
    .progress-seg.active {{
      background: color-mix(in srgb, var(--accent) 70%, #4ade80);
      animation: seg-pulse 1.2s ease-in-out infinite;
    }}
    .progress-seg.active::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.45) 50%, transparent 100%);
      animation: seg-shimmer 1.4s ease-in-out infinite;
    }}
    .progress-seg.done {{
      background: var(--accent);
    }}
    @keyframes seg-pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.75; }}
    }}
    @keyframes seg-shimmer {{
      0% {{ transform: translateX(-100%); }}
      100% {{ transform: translateX(100%); }}
    }}
    .live-run-card summary {{
      list-style: none;
      cursor: pointer;
    }}
    .live-run-card summary::-webkit-details-marker {{
      display: none;
    }}
    .live-run-body {{
      display: grid;
      gap: 12px;
      padding-top: 6px;
    }}
    .run-row {{
      min-width: 0;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(30, 36, 48, 0.08);
    }}
    .run-row strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font: 700 0.82rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .latest-step-details {{
      padding: 10px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(30, 36, 48, 0.08);
    }}
    .latest-step-details summary {{
      cursor: pointer;
      color: var(--muted);
      font: 700 0.82rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .latest-step-details summary::before {{
      content: "▸";
      font-size: 0.75rem;
      transition: transform 0.2s;
    }}
    .latest-step-details[open] summary::before {{
      transform: rotate(90deg);
    }}
    .latest-step-details summary::-webkit-details-marker {{
      display: none;
    }}
    .latest-step-content {{
      margin-top: 8px;
    }}
    .latest-action {{
      font: 500 0.9rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-word;
      margin-bottom: 6px;
    }}
    .latest-step-content .trace-meta {{
      padding: 0 0 6px;
    }}
    .latest-step-content .trace-shot {{
      padding: 6px 0 0;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 13px;
      background: rgba(15,118,110,0.12);
      color: var(--accent);
      font: 700 0.98rem/1 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
    }}
    .pulse {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: currentColor;
      box-shadow: 0 0 0 0 currentColor;
      animation: pulse 1.4s infinite;
    }}
    .mono {{
      font: 500 1rem/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-word;
    }}
    .trace-panel {{
    }}
    .trace-list {{
      display: grid;
      gap: 10px;
    }}
    .trace-item {{
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid rgba(30, 36, 48, 0.12);
      padding: 0;
      overflow: hidden;
    }}
    .trace-item summary {{
      list-style: none;
      cursor: pointer;
      padding: 14px 16px;
      font: 500 1rem/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-word;
    }}
    .trace-item summary::-webkit-details-marker {{
      display: none;
    }}
    .trace-item[open] summary {{
      border-bottom: 1px solid rgba(30, 36, 48, 0.08);
      background: rgba(255,255,255,0.55);
    }}
    .trace-meta {{
      padding: 12px 16px 0;
      color: var(--muted);
      font: 500 0.92rem/1.5 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      word-break: break-word;
    }}
    .trace-shot {{
      display: block;
      width: 100%;
      height: auto;
      padding: 10px 14px 14px;
      border-radius: 20px;
    }}
    .history {{
      display: grid;
      gap: 12px;
      max-height: calc(100vh - 120px);
      overflow: auto;
      padding-right: 4px;
    }}
    .drawer-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(16, 24, 40, 0.26);
      backdrop-filter: blur(3px);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
      z-index: 39;
    }}
    .drawer {{
      position: fixed;
      top: 0;
      right: 0;
      width: min(720px, calc(100vw - 16px));
      height: 100vh;
      padding: 16px;
      transform: translateX(104%);
      transition: transform 0.24s ease;
      z-index: 40;
    }}
    .drawer.open {{ transform: translateX(0); }}
    .drawer-backdrop.open {{
      opacity: 1;
      pointer-events: auto;
    }}
    .drawer-panel {{
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }}
    .drawer-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .record {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.88);
      display: grid;
      gap: 6px;
      cursor: pointer;
      transition: box-shadow 0.18s ease, border-color 0.18s ease;
      position: relative;
    }}
    .record:hover {{
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      border-color: var(--accent);
    }}
    .record .chevron {{
      display: inline-block;
      width: 18px;
      height: 18px;
      transition: transform 0.25s ease;
      color: var(--muted);
      flex-shrink: 0;
    }}
    .record.expanded .chevron {{
      transform: rotate(180deg);
    }}
    .record-header {{
      display: flex;
      align-items: center;
      gap: 8px;
      font: 600 0.82rem/1.2 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      color: var(--muted);
      letter-spacing: 0.03em;
    }}
    .record-header .record-meta {{
      flex: 1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .record-header .delete-btn {{
      background: none;
      border: none;
      cursor: pointer;
      color: var(--muted);
      padding: 2px 4px;
      font-size: 1rem;
      line-height: 1;
      border-radius: 6px;
      transition: color 0.15s, background 0.15s;
      flex-shrink: 0;
    }}
    .record-header .delete-btn:hover {{
      color: #dc2626;
      background: rgba(220,38,38,0.08);
    }}
    .record p {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .record-copy {{
      font-size: 0.92rem;
      line-height: 1.45;
      max-height: 2.9em;
      overflow: hidden;
      transition: max-height 0.25s ease;
    }}
    .record.expanded .record-copy {{
      max-height: none;
    }}
    .record .answer {{
      font: 600 0.95rem/1.45 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      max-height: 2.9em;
      overflow: hidden;
      transition: max-height 0.25s ease;
    }}
    .record.expanded .answer {{
      max-height: none;
    }}
    .record .meta-line {{
      color: var(--muted);
      font: 500 0.82rem/1.4 "Avenir Next", "Hiragino Sans", "Yu Gothic", sans-serif;
      max-height: 1.4em;
      overflow: hidden;
      transition: max-height 0.25s ease;
    }}
    .record.expanded .meta-line {{
      max-height: none;
    }}
    .record.expanded .preview {{
      display: none;
    }}
    .record.expanded .full {{
      display: inline !important;
    }}
    .record .full {{
      display: none;
    }}
    a {{ color: var(--accent); }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(15,118,110,0.45); }}
      70% {{ box-shadow: 0 0 0 10px rgba(15,118,110,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(15,118,110,0); }}
    }}
    @media (max-width: 960px) {{
      main {{ width: min(calc(100vw - 20px), 100%); }}
      .grid {{ grid-template-columns: 1fr; }}
      .history {{ max-height: none; }}
      .side-stack {{ position: static; }}
      .lang-switch {{
        width: 100%;
        justify-content: space-between;
      }}
      .lang-switch a {{
        flex: 1;
        text-align: center;
      }}
      .field-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-top">
        <div class="brand">
          <img class="brand-mark" src="/assets/logo-mark.png" alt="MolmoWeb icon">
          <h1>{escape(_t(lang, "title"))}</h1>
        </div>
        <div class="top-actions">
          <div class="help-wrap">
            <button id="help-toggle" class="secondary help-button" type="button">{escape(_t(lang, "help_button"))}</button>
            <div id="help-popover" class="help-popover">
              <h3 class="help-heading">{escape(_t(lang, "help_heading"))}</h3>
              <dl class="help-steps">
                {"".join(f'<div class="help-step"><dt><span class="step-num">{escape(num)}</span> {escape(title)}</dt><dd>{escape(desc)}</dd></div>' for num, title, desc in _t(lang, "help_steps"))}
              </dl>
              <p class="help-note">{escape(_t(lang, "help_note"))} <code>{escape(STATE.endpoint)}</code></p>
            </div>
          </div>
          <button id="history-toggle" class="secondary" type="button">{escape(_t(lang, "history_toggle"))}</button>
          <nav class="lang-switch" aria-label="Language switcher">
            <a href="/?lang=ja" class="{"active" if lang == "ja" else ""}">{escape(_t(lang, "lang_ja"))}</a>
            <a href="/?lang=en" class="{"active" if lang == "en" else ""}">{escape(_t(lang, "lang_en"))}</a>
          </nav>
        </div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <div class="meta">
          <div><strong>{escape(_t(lang, "browser_session"))}</strong><span id="browser-status">{escape(data["browser_status"])}</span></div>
          <div><strong>{escape(_t(lang, "headless"))}</strong>{"true" if STATE.headless else "false"}</div>
        </div>
        <form id="run-form">
          <input type="hidden" name="lang" value="{lang}">
          <label for="prompt">{escape(_t(lang, "task"))}</label>
          <textarea id="prompt" name="prompt" placeholder="{escape(_t(lang, "task_placeholder"))}"></textarea>
          <div class="field-grid">
            <div>
              <label for="max_steps">{escape(_t(lang, "max_steps"))}</label>
              <input id="max_steps" name="max_steps" type="number" min="1" max="50" value="{STATE.max_steps}">
            </div>
            <div class="field-group">
              <label for="window_width">{escape(_t(lang, "width"))}</label>
              <input id="window_width" name="window_width" type="number" min="640" value="{STATE.viewport_width}">
            </div>
            <div class="field-group">
              <label for="window_height">{escape(_t(lang, "height"))}</label>
              <input id="window_height" name="window_height" type="number" min="480" value="{STATE.viewport_height}">
            </div>
          </div>
          <div class="actions">
            <button id="run-button" class="primary" type="submit">
              <span class="spinner" aria-hidden="true"></span>
              <span id="run-button-label">{escape(_t(lang, "run_task"))}</span>
            </button>
            <button id="reset-button" class="secondary" type="button">{escape(_t(lang, "reset"))}</button>
            <button id="stop-button" class="secondary" type="button" disabled>{escape(_t(lang, "force_stop"))}</button>
          </div>
        </form>
        <p id="notice" class="notice" style="display:none;"></p>
        <p id="error" class="error" style="display:none;"></p>
      </div>
      <div class="side-stack">
        <details class="panel live-run live-run-card" open>
          <summary class="live-head">
            <span>{escape(_t(lang, "current_run"))}</span>
            <span class="badge"><span class="pulse"></span><span id="live-status">{escape(_t(lang, "idle"))}</span></span>
          </summary>
          <div class="progress-bar" id="progress-bar"></div>
          <div class="live-run-body">
            <div class="run-row">
              <strong>{escape(_t(lang, "page_title"))}</strong>
              <div id="live-page-title">{escape(data["page_title"] or "-")}</div>
            </div>
            <details class="latest-step-details" open>
              <summary>{escape(_t(lang, "current_step"))}</summary>
              <div class="latest-step-content" id="latest-step">{data["latest_step_html"]}</div>
            </details>
          </div>
        </details>
        <div class="panel trace-panel">
          <div class="live-head">
            <span>{escape(_t(lang, "live_trace"))}</span>
            <div class="live-head-actions">
              <label class="toggle-switch-inline">
                <span class="toggle-label">{escape(_t(lang, "expand_all"))}</span>
                <input type="checkbox" id="trace-toggle" checked>
                <span class="toggle-slider"></span>
              </label>
              <span class="badge"><span id="live-step-badge">{current_step}</span></span>
            </div>
          </div>
          <div class="trace-list" id="trace-list">{data["trace_html"]}</div>
        </div>
      </div>
    </section>
  </main>
  <div id="history-backdrop" class="drawer-backdrop"></div>
  <aside id="history-drawer" class="drawer" aria-hidden="true">
    <div class="panel drawer-panel">
      <div class="drawer-head">
        <h2 style="margin:0;">{escape(_t(lang, "history"))}</h2>
        <div class="top-actions">
          <button id="history-clear" class="secondary" type="button">{escape(_t(lang, "clear_history"))}</button>
          <button id="history-close" class="secondary" type="button">{escape(_t(lang, "history_close"))}</button>
        </div>
      </div>
      <div class="history" id="history">{data["history_html"]}</div>
    </div>
  </aside>
  <script>
    const lang = {lang!r};
    const text = {{
      runTask: {_t(lang, "run_task")!r},
      running: {_t(lang, "running")!r},
      forceStop: {_t(lang, "force_stop")!r},
      taskFinished: {_t(lang, "task_finished")!r},
      taskFailed: {_t(lang, "task_failed")!r},
      enterTask: {_t(lang, "enter_task")!r},
      idle: {_t(lang, "idle")!r},
      queued: {_t(lang, "queued")!r},
    }};
    const form = document.getElementById("run-form");
    const runButton = document.getElementById("run-button");
    const runButtonLabel = document.getElementById("run-button-label");
    const resetButton = document.getElementById("reset-button");
    const stopButton = document.getElementById("stop-button");
    const historyToggle = document.getElementById("history-toggle");
    const helpToggle = document.getElementById("help-toggle");
    const helpPopover = document.getElementById("help-popover");
    const historyClose = document.getElementById("history-close");
    const historyClear = document.getElementById("history-clear");
    const historyDrawer = document.getElementById("history-drawer");
    const historyBackdrop = document.getElementById("history-backdrop");
    const notice = document.getElementById("notice");
    const error = document.getElementById("error");
    const traceToggle = document.getElementById("trace-toggle");
    let pollTimer = null;

    function applyTraceToggle() {{
      const open = traceToggle.checked;
      document.querySelectorAll(".trace-item").forEach(el => {{
        el.open = open;
      }});
    }}
    traceToggle.addEventListener("change", applyTraceToggle);

    function setDrawer(open) {{
      historyDrawer.classList.toggle("open", open);
      historyBackdrop.classList.toggle("open", open);
      historyDrawer.setAttribute("aria-hidden", open ? "false" : "true");
    }}

    function setHelp(open) {{
      helpPopover.classList.toggle("open", open);
    }}

    function setRunningUI(isRunning) {{
      runButton.disabled = isRunning;
      resetButton.disabled = isRunning;
      stopButton.disabled = !isRunning;
      runButton.classList.toggle("is-running", isRunning);
      runButtonLabel.textContent = isRunning ? text.running : text.runTask;
    }}

    function setMessage(target, value) {{
      if (!value) {{
        target.style.display = "none";
        target.textContent = "";
        return;
      }}
      target.style.display = "block";
      target.textContent = value;
    }}

    function updateProgressBar(data) {{
      const bar = document.getElementById("progress-bar");
      const maxSteps = data.max_steps || 0;
      const stepNum = data.step_num || 0;
      const isRunning = Boolean(data.running);
      const isDone = !isRunning && data.status !== "idle";
      if (!maxSteps) {{
        bar.innerHTML = "";
        return;
      }}
      // Build segments if count changed
      if (bar.children.length !== maxSteps) {{
        bar.innerHTML = Array.from({{length: maxSteps}}, () => '<div class="progress-seg"></div>').join("");
      }}
      const segs = bar.children;
      for (let i = 0; i < maxSteps; i++) {{
        if (isDone) {{
          segs[i].className = "progress-seg done";
        }} else if (i < stepNum - 1) {{
          segs[i].className = "progress-seg filled";
        }} else if (i === stepNum - 1 && isRunning) {{
          segs[i].className = "progress-seg active";
        }} else if (i < stepNum) {{
          segs[i].className = "progress-seg filled";
        }} else {{
          segs[i].className = "progress-seg";
        }}
      }}
    }}

    function updateLive(data) {{
      document.getElementById("browser-status").textContent = data.browser_status;
      document.getElementById("live-status").textContent = data.localized_status;
      document.getElementById("live-page-title").textContent = data.page_title || "-";
      document.getElementById("latest-step").innerHTML = data.latest_step_html || "-";
      document.getElementById("trace-list").innerHTML = data.trace_html;
      applyTraceToggle();
      document.getElementById("history").innerHTML = data.history_html;
      attachRecordListeners();
      const step = data.step_num ? `${{data.step_num}}/${{data.max_steps}}` : (data.running ? text.queued : "-");
      document.getElementById("live-step-badge").textContent = step;
      updateProgressBar(data);
      setRunningUI(Boolean(data.running));
      setMessage(error, data.last_error || data.error || "");
      if (!data.running && data.status !== "idle") {{
        setMessage(notice, data.error ? text.taskFailed : text.taskFinished);
      }}
    }}

    async function fetchStatus() {{
      const response = await fetch(`/api/status?lang=${{encodeURIComponent(lang)}}`, {{ cache: "no-store" }});
      const data = await response.json();
      updateLive(data);
      if (!data.running && pollTimer) {{
        window.clearInterval(pollTimer);
        pollTimer = null;
      }}
    }}

    function parseWindowSize(widthValue, heightValue) {{
      const width = Number(widthValue);
      const height = Number(heightValue);
      if (!Number.isFinite(width) || !Number.isFinite(height) || width < 640 || height < 480) return null;
      return {{ width, height }};
    }}

    async function deleteHistoryRecord(recordId) {{
      const formData = new FormData();
      formData.append("lang", lang);
      const response = await fetch(`/api/history/${{encodeURIComponent(recordId)}}/delete`, {{
        method: "POST",
        body: formData,
      }});
      const data = await response.json();
      updateLive(data);
    }}

    async function clearHistory() {{
      const formData = new FormData();
      formData.append("lang", lang);
      const response = await fetch("/api/history/clear", {{
        method: "POST",
        body: formData,
      }});
      const data = await response.json();
      updateLive(data);
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      setMessage(notice, "");
      setMessage(error, "");
      const prompt = document.getElementById("prompt").value.trim();
      if (!prompt) {{
        setMessage(error, text.enterTask);
        return;
      }}
      const size = parseWindowSize(
        document.getElementById("window_width").value,
        document.getElementById("window_height").value,
      );
      if (!size) {{
        setMessage(error, {_t(lang, "window_size_invalid")!r});
        return;
      }}
      setRunningUI(true);
      document.getElementById("live-step-badge").textContent = text.queued;
      document.getElementById("live-page-title").textContent = "-";
      document.getElementById("latest-step").innerHTML = "-";
      document.getElementById("trace-list").innerHTML = `<p class="hint">${{text.queued}}</p>`;
      document.getElementById("live-status").textContent = text.running;
      const formData = new FormData(form);
      formData.set("window_width", String(size.width));
      formData.set("window_height", String(size.height));
      const response = await fetch("/api/run", {{
        method: "POST",
        body: formData,
      }});
      const data = await response.json();
      if (!response.ok) {{
        setRunningUI(false);
        setMessage(error, data.detail || text.taskFailed);
        return;
      }}
      updateLive(data);
      if (pollTimer) window.clearInterval(pollTimer);
      pollTimer = window.setInterval(fetchStatus, 1000);
    }});

    // Persist prompt across language switches
    const promptEl = document.getElementById("prompt");
    const saved = sessionStorage.getItem("molmoweb_prompt");
    if (saved) promptEl.value = saved;
    promptEl.addEventListener("input", () => {{
      sessionStorage.setItem("molmoweb_prompt", promptEl.value);
    }});

    fetchStatus();
    helpToggle.addEventListener("click", () => setHelp(!helpPopover.classList.contains("open")));
    historyToggle.addEventListener("click", () => setDrawer(true));
    historyClose.addEventListener("click", () => setDrawer(false));
    historyBackdrop.addEventListener("click", () => setDrawer(false));
    document.addEventListener("click", (event) => {{
      if (!helpPopover.classList.contains("open")) return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (helpPopover.contains(target) || helpToggle.contains(target)) return;
      setHelp(false);
    }});
    historyClear.addEventListener("click", clearHistory);
    resetButton.addEventListener("click", async () => {{
      if (runButton.disabled) return;
      setMessage(notice, "");
      setMessage(error, "");
      const formData = new FormData();
      formData.append("lang", lang);
      const response = await fetch("/reset", {{
        method: "POST",
        body: formData,
        redirect: "follow",
      }});
      if (!response.ok) {{
        setMessage(error, text.taskFailed);
        return;
      }}
      await fetchStatus();
    }});
    stopButton.addEventListener("click", async () => {{
      if (stopButton.disabled) return;
      setMessage(notice, "");
      setMessage(error, "");
      const formData = new FormData();
      formData.append("lang", lang);
      const response = await fetch("/api/stop", {{
        method: "POST",
        body: formData,
      }});
      const data = await response.json();
      updateLive(data);
    }});
    // Handle delete buttons and prevent card toggle
    function attachRecordListeners() {{
      document.querySelectorAll(".record .delete-btn").forEach(btn => {{
        btn.onclick = async (e) => {{
          e.stopPropagation();
          await deleteHistoryRecord(btn.dataset.recordId);
        }};
      }});
      document.querySelectorAll(".record a").forEach(el => {{
        el.addEventListener("click", (e) => e.stopPropagation());
      }});
    }}
    attachRecordListeners();
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
def index(lang: str = "en") -> HTMLResponse:
    return _render_page(lang=lang)


@app.get("/api/status")
def api_status(lang: str = "en") -> JSONResponse:
    with STATE.lock:
        return JSONResponse(_status_payload(_lang_value(lang)))


@app.post("/api/run")
def api_run(
    prompt: str = Form(...),
    max_steps: int = Form(DEFAULT_MAX_STEPS),
    lang: str = Form("en"),
    window_width: int = Form(1600),
    window_height: int = Form(1000),
) -> JSONResponse:
    lang = _lang_value(lang)
    prompt = prompt.strip()
    if not prompt:
        return JSONResponse({"detail": _t(lang, "enter_task")}, status_code=400)
    width = int(window_width)
    height = int(window_height)
    if width < 640 or height < 480:
        return JSONResponse(
            {"detail": _t(lang, "window_size_invalid")}, status_code=400
        )

    with STATE.lock:
        if STATE.live_run.running:
            return JSONResponse({"detail": _t(lang, "running")}, status_code=409)
        if STATE.viewport_width != width or STATE.viewport_height != height:
            STATE.viewport_width = width
            STATE.viewport_height = height
            if STATE.client is not None:
                STATE.client.close()
                STATE.client = None
        STATE.last_error = None
        STATE.stop_requested = False
        STATE.live_run = LiveRun(
            running=True,
            prompt=prompt,
            max_steps=max_steps,
            status="running",
            started_at=datetime.now().strftime("%H:%M:%S"),
            steps_log=[],
        )
        STATE.ensure_worker()
        STATE.task_queue.put((prompt, max_steps))
        return JSONResponse(_status_payload(lang))


@app.post("/reset")
def reset(lang: str = Form("en")) -> RedirectResponse:
    with STATE.lock:
        if not STATE.live_run.running:
            STATE.reset_browser()
    return RedirectResponse(f"/?lang={_lang_value(lang)}", status_code=303)


@app.post("/api/stop")
def api_stop(lang: str = Form("en")) -> JSONResponse:
    lang = _lang_value(lang)
    with STATE.lock:
        if not STATE.live_run.running:
            return JSONResponse(_status_payload(lang))
        STATE.force_stop()
        return JSONResponse(_status_payload(lang))


@app.post("/api/history/{record_id}/delete")
def delete_history_record(record_id: str, lang: str = Form("en")) -> JSONResponse:
    with STATE.lock:
        STATE.history = [record for record in STATE.history if record.id != record_id]
        _save_history()
        return JSONResponse(_status_payload(_lang_value(lang)))


@app.post("/api/history/clear")
def clear_history(lang: str = Form("en")) -> JSONResponse:
    with STATE.lock:
        STATE.history = []
        _save_history()
        return JSONResponse(_status_payload(_lang_value(lang)))
