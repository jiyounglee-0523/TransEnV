# Trans-EnV Live Demo

A Gradio app that transforms Standard American English (SAE) into an English
variety by running the Trans-EnV guideline pipeline rule by rule:

- **Dialects** — 18 varieties driven by their eWAVE linguistic features
  (5–36 rules per dialect).
- **ESL learner English** — 10 native languages × CEFR level A/B: 10 randomly
  sampled CEFR simplification rules (English Grammar Profile) applied first,
  then the language's L1 grammar-transfer error rules. Level "None" applies
  the L1 errors only. The full benchmark pipeline applies all 506 (A) / 146 (B)
  CEFR rules offline (`src/run/main.py --task_name cefr`); the demo samples
  them for interactive speed.

Each rule is checked and applied by an LLM one at a time, with an optional
semantic-preservation judge, a live rule-by-rule log, and a word-level diff
of the final result against the original.

Visitors choose a provider — OpenAI, Google Gemini, or Anthropic Claude — and
enter their own API key. The deployed Hugging Face Space additionally offers a
free ZeroGPU tier (Qwen2.5-7B-Instruct, no key needed):
https://huggingface.co/spaces/jiyounglee0523/Trans-EnV-demo

## Run locally

```bash
python -m venv demo/.venv
demo/.venv/bin/pip install -r demo/requirements.txt
demo/.venv/bin/python demo/app.py            # http://localhost:7860
demo/.venv/bin/python demo/app.py --share    # + temporary public link
```

## Layout

- `app.py` — local Gradio app (uses the repo's `assets/` and `src/`).
- `hf_space/` — self-contained copy deployed to the Hugging Face Space
  (adds the ZeroGPU free-trial provider). Redeploy with
  `huggingface_hub.upload_folder(folder_path='demo/hf_space', repo_id='jiyounglee0523/Trans-EnV-demo', repo_type='space')`.
- `assets/cefr_features.json` (repo root) — precomputed CEFR feature lists per
  level, derived from `assets/EnglishGrammarProfileOnline.xlsx`.
