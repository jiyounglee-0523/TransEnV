---
title: Trans-EnV Demo
emoji: 🌍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
license: mit
---

# Trans-EnV Live Demo

Transform Standard American English (SAE) into an English variety by running
the Trans-EnV guideline pipeline rule by rule:

- **Dialects** — 18 varieties (AAVE, Scottish, Irish, Australian, …) driven by
  their eWAVE linguistic features (5–36 rules per dialect).
- **ESL learner English** — 10 native languages (Arabic, Japanese, Mandarin, …)
  × CEFR level A/B: 10 randomly sampled CEFR simplification rules (English
  Grammar Profile) applied first, then the language's grammar-transfer error
  rules. Level "None" applies the L1 errors only.

Each rule is checked and applied by an LLM one at a time, and an optional
semantic-preservation judge rejects rewrites that lose meaning. The rule-by-rule
log streams live, and the final result is shown as a word-level diff against
the original sentence.

**Models** — try it free on ZeroGPU (Qwen2.5-7B-Instruct, no key needed), or
pick OpenAI (gpt-4o-mini, gpt-5.6 family, …), Google Gemini (gemini-3.7-flash,
…), or Anthropic Claude (claude-opus-5, …) and enter your own API key. Keys are
used only for your session's requests and are never stored.

Note: this demo samples the CEFR stage for interactive speed — the full
benchmark pipeline applies all 506 (level A) / 146 (level B) rules offline.

Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs
Against English Varieties.
