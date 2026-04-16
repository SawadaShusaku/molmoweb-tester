# MolmoWeb Tester

Local testing GUI for MolmoWeb on macOS.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Platform](#platform)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the GUI](#using-the-gui)
- [Project Layout](#project-layout)
- [Notes](#notes)

## Overview

MolmoWeb Tester adds a local browser UI on top of MolmoWeb so you can:

- send natural-language tasks
- monitor the current run
- inspect execution steps with screenshots
- reopen previous runs from history

This repository is currently tuned for macOS, especially Apple Silicon Macs.

## Features

- FastAPI GUI for local testing
- Live run status and step-by-step trace
- Persistent local history
- Resettable Chromium session
- Japanese / English UI switcher
- Custom browser width and height
- `mps` and CPU fallback for non-CUDA Macs

## Platform

Current target:

- macOS
- Apple Silicon recommended

Possible future extension:

- Windows support

## Requirements

- Python 3.10+
- `uv`
- Playwright Chromium
- MolmoWeb checkpoint access

## Installation

```bash
uv sync
uv run playwright install chromium
```

## Quick Start

Start the model server:

```bash
cd /path/to/molmoweb
export PREDICTOR_TYPE=hf
bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B
```

Start the GUI in another terminal:

```bash
cd /path/to/molmoweb
bash scripts/start_gui.sh 8010
```

Open the tester:

```text
http://127.0.0.1:8010/?lang=ja
```

## Using the GUI

1. Enter a task in the main textarea.
2. Set the maximum step count.
3. Set Chromium width and height.
4. Click `Run task`.
5. Watch the right-side panels for live status and execution steps.
6. Open `History` to review past runs.
7. Use `Reset browser session` when you want a fresh Chromium window.

## Project Layout

- `agent/`
  Model server and predictor backends
- `inference/client.py`
  MolmoWeb client and browser session handling
- `inference/gui_app.py`
  Local tester GUI
- `scripts/start_server.sh`
  Model server launcher
- `scripts/start_gui.sh`
  GUI launcher
- `assets/`
  Logo and README images

## Notes

- Chromium runs in a separate window.
- This tester does not control your current Safari or Brave session.
- Some interactions like right-click and tab handling are still imperfect.
- History files are stored locally under `inference/htmls/gui/`.
