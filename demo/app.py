"""Trans-EnV live demo.

A Gradio app that transforms Standard American English (SAE) text into a
selected English variety (dialect or ESL learner English) by applying the
Trans-EnV guideline pipeline rule by rule with a user-provided LLM API key.

Run:
    python demo/app.py [--port 7860] [--share]
"""

import argparse
import difflib
import html
import json
import os
import random
import re
import sys
from pathlib import Path

import anthropic
import gradio as gr
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from registry.guidline import L1_GRAMMARERROR                      # noqa: E402
from registry.prompt import return_system_message, semantic_check  # noqa: E402
from utils.guidline_utils import (                                 # noqa: E402
    extract_guideline_examples,
    extract_transformed_sentence,
)

# ---------------------------------------------------------------------------
# Guideline loading
# ---------------------------------------------------------------------------

DIALECTS = {
    "AAVE (Urban African American Vernacular English)": "Urban African American Vernacular English",
    "Appalachian English": "Appalachian English",
    "Australian English": "Australian English",
    "Australian Vernacular English": "Australian Vernacular English",
    "Bahamian English": "Bahamian English",
    "East Anglian English": "East Anglian English",
    "Irish English": "Irish English",
    "Manx English": "Manx English",
    "New Zealand English": "New Zealand English",
    "Newfoundland English": "Newfoundland English",
    "North of England dialects": "English dialects in the North of England",
    "Ozark English": "Ozark English",
    "Scottish English": "Scottish English",
    "Southeast American enclave dialects": "Southeast American enclave dialects",
    "Southeast of England dialects": "English dialects in the Southeast of England",
    "Southwest of England dialects": "English dialects in the Southwest of England",
    "Tristan da Cunha English": "Tristan da Cunha English",
    "Welsh English": "Welsh English",
}

L1_LANGUAGES = list(L1_GRAMMARERROR.keys())

with open(ROOT / "assets/guidelines/orig_generated_guideline_wo_example.json") as f:
    _dialect_guidelines = {g["feature"][3:-3]: g["guideline"] for g in json.load(f)}

with open(ROOT / "assets/guidelines/python_grammar_error.json") as f:
    _l1_guidelines = {g["grammar_error"]: g["guideline"] for g in json.load(f)}

with open(ROOT / "assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json") as f:
    _cefr_guidelines = {g["feature"][1:-1].strip(): g["guideline"] for g in json.load(f)}

with open(ROOT / "assets/cefr_features.json") as f:
    _cefr_features = json.load(f)  # {"A": [...], "B": [...]} — precomputed from the Grammar Profile

_ewave = pd.read_csv(ROOT / "assets/ewave/ewave.csv")

CEFR_SAMPLE_SIZE = 10
CEFR_CHOICES = ["A (beginner)", "B (intermediate)", "None (L1 errors only)"]


def build_guideline(variety_type, dialect_label, l1_language, cefr_level):
    """Return stages: a list of lists of (feature, guideline_text, task_name)."""
    if variety_type == "Dialect":
        language_id = DIALECTS[dialect_label]
        features = _ewave[
            (_ewave["Language_ID"] == language_id) & (_ewave["Value"] == "A")
        ]["Parameter_ID"].tolist()
        return [[(f, _dialect_guidelines[f], "english_dialect")
                 for f in features if f in _dialect_guidelines]]

    stages = []
    if cefr_level and cefr_level[0] in "AB":
        level = cefr_level[0]
        sampled = random.sample(
            _cefr_features[level], min(CEFR_SAMPLE_SIZE, len(_cefr_features[level]))
        )
        stages.append([(f"CEFR-{level}: {f}", _cefr_guidelines[f], "cefr")
                       for f in sampled])
    features = L1_GRAMMARERROR[l1_language]
    stages.append([(f, _l1_guidelines[f], "L1")
                   for f in features if f in _l1_guidelines])
    return stages


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

PROVIDERS = {
    "OpenAI": {
        "base_url": None,
        "models": ["gpt-4o-mini", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol",
                   "gpt-5.4-mini", "gpt-5.2", "gpt-4o", "gpt-4.1"],
        "default": "gpt-4o-mini",
        "key_hint": "sk-...",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                   "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro",
                   "gemini-flash-latest"],
        "default": "gemini-3.7-flash",
        "key_hint": "AIza...",
    },
    "Anthropic Claude": {
        "sdk": "anthropic",
        "models": ["claude-opus-5", "claude-sonnet-5", "claude-sonnet-4-6",
                   "claude-haiku-4-5", "claude-opus-4-8"],
        "default": "claude-opus-5",
        "key_hint": "sk-ant-...",
    },
}


def make_client(provider, api_key):
    if PROVIDERS[provider].get("sdk") == "anthropic":
        return anthropic.Anthropic(api_key=api_key.strip())
    return OpenAI(api_key=api_key.strip(), base_url=PROVIDERS[provider]["base_url"])


def chat(client, model, messages, temperature=0.8, top_p=0.95, max_tokens=2000):
    if isinstance(client, anthropic.Anthropic):
        # Claude 4.6+ models take no temperature/top_p; adaptive thinking runs by
        # default and its tokens count toward max_tokens, so leave headroom.
        system = None
        converted = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                converted.append(m)
        kwargs = {"model": model, "max_tokens": max_tokens * 4, "messages": converted}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return "".join(b.text for b in response.content if b.type == "text")
    if model.startswith("gpt-5"):
        # gpt-5.x reasoning models: no max_tokens/temperature/top_p; leave
        # headroom for reasoning tokens on top of the visible answer.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens * 4,
        )
        return response.choices[0].message.content
    if model.startswith(("gemini-3", "gemini-flash-latest")):
        # Thinking models: thoughts count toward the output token limit.
        max_tokens = max_tokens * 4
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Newer OpenAI models reject max_tokens / non-default temperature.
        if "max_tokens" in str(e) or "temperature" in str(e):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            raise
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Transformation pipeline (chat-API port of src/framework/transformation.py)
# ---------------------------------------------------------------------------

def build_messages(feature, guideline_text, task_name, sentence):
    guideline_instruction, example = extract_guideline_examples(guideline_text, task_name)
    system_message = return_system_message(guideline_instruction)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example[0]["input"]},
        {"role": "assistant", "content": example[0]["output"]},
        {"role": "user", "content": f"**Original Sentence:** {sentence}"},
    ]


def strip_wrapping_quotes(candidate):
    """Drop a single pair of double quotes wrapping the whole sentence."""
    result = candidate.strip()
    if len(result) >= 2 and result[0] in '"“' and result[-1] in '"”':
        result = result[1:-1].strip()
    return result


def word_diff(original, transformed):
    """Return (original_html, transformed_html) with word-level change marks."""
    a = re.findall(r"\S+|\s+", original)
    b = re.findall(r"\S+|\s+", transformed)
    sm = difflib.SequenceMatcher(None, a, b)
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a = html.escape("".join(a[i1:i2]))
        seg_b = html.escape("".join(b[j1:j2]))
        if op == "equal":
            out_a.append(seg_a)
            out_b.append(seg_b)
        else:
            if seg_a.strip():
                out_a.append(f'<mark class="del">{seg_a}</mark>')
            else:
                out_a.append(seg_a)
            if seg_b.strip():
                out_b.append(f'<mark class="add">{seg_b}</mark>')
            else:
                out_b.append(seg_b)
    return "".join(out_a), "".join(out_b)


# ---------------------------------------------------------------------------
# Examples gallery (pre-transformed GSM8K questions from the released benchmark)
# ---------------------------------------------------------------------------

with open(ROOT / "assets/gallery_examples.json") as f:
    _gallery = json.load(f)  # {"dialect"|"l1": {variety_key: [{"o":…, "t":…} × 8]}}

GALLERY_DIALECT_KEYS = {
    "AAVE (Urban African American Vernacular English)": "aave",
    "Appalachian English": "appalachian",
    "Australian English": "australian",
    "Australian Vernacular English": "australian_vernacular",
    "Bahamian English": "bhamanian",
    "East Anglian English": "east_anglian",
    "Irish English": "irish",
    "Manx English": "manx",
    "New Zealand English": "new_zealand",
    "Newfoundland English": "newfoundland",
    "North of England dialects": "north_england",
    "Ozark English": "ozark",
    "Scottish English": "scottish",
    "Southeast American enclave dialects": "southeast_american",
    "Southeast of England dialects": "southeast_england",
    "Southwest of England dialects": "southwest_england",
    "Tristan da Cunha English": "cunha",
    "Welsh English": "welsh",
}

_L1_LANG_NAMES = {
    "arabic": "Arabic", "chinese_mandarin": "Chinese-Mandarin", "french": "French",
    "german": "German", "italian": "Italian", "japanese": "Japanese",
    "portuguese": "Portuguese", "russian": "Russian", "spanish": "Spanish",
    "turkish": "Turkish",
}
GALLERY_L1_KEYS = {}
for _key in sorted(_gallery["l1"]):
    _level, _lang = _key.split("_", 1)
    GALLERY_L1_KEYS[f"{_L1_LANG_NAMES.get(_lang, _lang.title())} (CEFR {_level})"] = _key


def gallery_pairs(gallery_type, variety_label):
    if gallery_type == "Dialect":
        return _gallery["dialect"][GALLERY_DIALECT_KEYS[variety_label]]
    return _gallery["l1"][GALLERY_L1_KEYS[variety_label]]


def gallery_choices(gallery_type, variety_label):
    pairs = gallery_pairs(gallery_type, variety_label)
    return [f"{i + 1}. {p['o'][:70].rstrip()}…" for i, p in enumerate(pairs)]


def render_gallery(gallery_type, variety_label, example_choice):
    pairs = gallery_pairs(gallery_type, variety_label)
    i = int(str(example_choice).split(".", 1)[0]) - 1
    i = max(0, min(i, len(pairs) - 1))
    original, transformed = pairs[i]["o"], pairs[i]["t"]
    orig_html, trans_html = word_diff(original, transformed)
    notice = ""
    return f"""
    <div class="panes">
      <div class="pane">
        <p class="panelabel">Standard American English</p>
        <p class="sentence">{orig_html}</p>
      </div>
      <div class="pane">
        <p class="panelabel">{html.escape(variety_label)}</p>
        <p class="sentence">{trans_html}</p>
        {notice}
      </div>
    </div>
    """


STATUS_ICONS = {
    "applied": ("applied", "&#10003;"),
    "nochange": ("nochange", "&ndash;"),
    "rejected": ("rejected", "&#8856;"),
    "error": ("error", "!"),
}


def render_log(entries, total):
    rows = []
    for feature, status, note in entries:
        cls, icon = STATUS_ICONS[status]
        note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
        rows.append(
            f'<li class="{cls}"><span class="icon">{icon}</span>'
            f"<span class='feat'>{html.escape(feature)}</span>{note_html}</li>"
        )
    return (
        f'<div class="rulelog"><p class="logcount">{len(entries)} / {total} rules checked</p>'
        f'<ul>{"".join(rows)}</ul></div>'
    )


def render_result(original, current, applied, done=False):
    if done:
        orig_html, trans_html = word_diff(original, current)
        badge = '<span class="badge done">Final</span>'
        chips = "".join(f'<span class="chip">{html.escape(r)}</span>' for r in applied)
        chips_block = f'<div class="chips">{chips}</div>' if chips else ""
        right = f'<p class="sentence">{trans_html}</p>{chips_block}'
    else:
        # Hide the work-in-progress text until every rule has been checked.
        orig_html = html.escape(original)
        badge = '<span class="badge running">Transforming&hellip;</span>'
        right = ('<p class="sentence pending">The transformed sentence will appear '
                 "once all rules have been checked.</p>")
    return f"""
    <div class="panes">
      <div class="pane">
        <p class="panelabel">Standard American English</p>
        <p class="sentence">{orig_html}</p>
      </div>
      <div class="pane">
        <p class="panelabel">Transformed {badge}</p>
        {right}
      </div>
    </div>
    """


def transform(text, variety_type, dialect_label, l1_language, cefr_level,
              provider, model, api_key, shuffle_rules, use_judge):
    text = (text or "").strip()
    if not text:
        yield "", "", "Enter a sentence to transform."
        return
    if not (api_key or "").strip():
        yield "", "", f"Enter your {provider} API key ({PROVIDERS[provider]['key_hint']})."
        return

    stages = build_guideline(variety_type, dialect_label, l1_language, cefr_level)
    guideline = []
    for stage in stages:
        stage = stage[:]
        if shuffle_rules:
            random.shuffle(stage)
        guideline.extend(stage)

    client = make_client(provider, api_key)
    sentence = re.sub(r"_{2,}", "<blank>", text)
    original = sentence
    total = len(guideline)
    applied, log_entries = [], []
    errors = 0

    yield render_result(original, sentence, applied), render_log(log_entries, total), \
        f"Applying {total} rules with {model}&hellip;"

    for feature, guideline_text, task_name in guideline:
        try:
            messages = build_messages(feature, guideline_text, task_name, sentence)
            response = chat(client, model, messages)
            candidate = strip_wrapping_quotes(
                extract_transformed_sentence(response or "")
            )

            if "no change" in candidate.lower():
                log_entries.append((feature, "nochange", ""))
            elif use_judge:
                verdict = chat(
                    client, model,
                    [{"role": "user", "content": semantic_check(original, candidate)}],
                )
                if "no" in (verdict or "").lower():
                    sentence = candidate
                    applied.append(feature)
                    log_entries.append((feature, "applied", ""))
                else:
                    log_entries.append((feature, "rejected", "meaning changed"))
            else:
                sentence = candidate
                applied.append(feature)
                log_entries.append((feature, "applied", ""))
        except Exception as e:
            errors += 1
            message = str(e)
            log_entries.append((feature, "error", message[:120]))
            lowered = message.lower()
            if any(k in lowered for k in ("api key", "authentication", "401", "permission")):
                yield render_result(original, sentence, applied), \
                    render_log(log_entries, total), \
                    f"Stopped: authentication failed &mdash; {html.escape(message[:200])}"
                return

        yield render_result(original, sentence, applied), render_log(log_entries, total), \
            f"Applying {total} rules with {model}&hellip; ({len(log_entries)}/{total})"

    final_sentence = sentence.replace("<blank>", "____")
    status = f"Done &mdash; {len(applied)} of {total} rules applied."
    if errors:
        status += f" ({errors} rule{'s' if errors > 1 else ''} skipped due to API errors.)"
    yield render_result(original, final_sentence, applied, done=True), \
        render_log(log_entries, total), status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
:root { --add-bg: #d9f2e5; --add-ink: #0b5e3c; --del-bg: #fbe3e0; --del-ink: #8f2f22; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#title h1 { margin-bottom: 0.1em; }
#title p { color: var(--body-text-color); margin-top: 0; }
.panes { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 800px) { .panes { grid-template-columns: 1fr; } }
.pane { border: 1px solid var(--border-color-primary); border-radius: 10px;
        padding: 14px 16px; background: var(--background-fill-primary); }
.panelabel { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
             color: var(--body-text-color-subdued); margin: 0 0 8px; }
.sentence { font-size: 1.05rem; line-height: 1.65; margin: 0; }
.sentence.pending { color: var(--body-text-color-subdued); font-style: italic; }
mark.add { background: #b7ecd2; color: #084c30; font-weight: 600;
           border-radius: 3px; padding: 0 3px; }
mark.del { background: #fbd5d0; color: #7f1d1d; border-radius: 3px; padding: 0 3px;
           text-decoration: line-through; text-decoration-thickness: 1.5px; }
.dark mark.add { background: #14532d; color: #bbf7d0; }
.dark mark.del { background: #641e1e; color: #fecaca; }
.badge { font-size: 0.7rem; border-radius: 999px; padding: 1px 8px; margin-left: 6px; }
.badge.done { background: var(--add-bg); color: var(--add-ink); }
.badge.running { background: var(--background-fill-secondary); }
.chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 0.72rem; border: 1px solid var(--border-color-primary);
        border-radius: 999px; padding: 2px 9px; color: var(--body-text-color-subdued); }
.rulelog ul { list-style: none; margin: 0; padding: 0; max-height: 340px; overflow-y: auto; }
.rulelog li { display: flex; gap: 8px; align-items: baseline; padding: 3px 0;
              font-size: 0.85rem; border-bottom: 1px dotted var(--border-color-primary); }
.rulelog .icon { width: 1.2em; text-align: center; font-weight: 700; }
.rulelog li.applied .icon { color: #0b7a4b; }
.rulelog li.nochange { color: var(--body-text-color-subdued); }
.rulelog li.rejected .icon { color: #b3591f; }
.rulelog li.error .icon { color: #a3342a; }
.rulelog .note { color: var(--body-text-color-subdued); font-size: 0.78rem; }
.logcount { font-size: 0.78rem; color: var(--body-text-color-subdued); margin: 0 0 6px; }
"""

EXAMPLES = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "The committee has not yet decided where the annual conference will be held next year.",
]


THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.teal,
    font=[gr.themes.GoogleFont("Archivo"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def build_app():
    with gr.Blocks(title="Trans-EnV Demo") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(
            "<div id='title'><h1>Trans-EnV</h1>"
            "<b>Trans-EnV: A Framework for Evaluating the Linguistic Robustness of "
            "LLMs Against English Varieties</b> (NeurIPS 2025 Datasets and Benchmarks "
            "Track) &mdash; <a href='https://arxiv.org/abs/2505.20875' "
            "target='_blank'>Paper</a> &middot; "
            "<a href='https://github.com/jiyounglee-0523/TransEnV' "
            "target='_blank'>Code</a><br>"
            "<p>Transform Standard American English into an English variety with the "
            "Trans-EnV guideline pipeline: 18 English dialects or English as Second Language (ESL) "
            "learner English.<br>"
            "Bring your own OpenAI, Gemini, or Anthropic API key.<br>"
            "\U0001F512 API keys are sent only to the provider you choose and are never stored or logged &mdash; feel free "
            "to use your own key with confidence.</p></div>"
        )

        with gr.Tabs():
            with gr.Tab("Live transform"):
                with gr.Row():
                    with gr.Column(scale=1):
                        provider = gr.Radio(list(PROVIDERS), value="OpenAI", label="Provider")
                        model = gr.Dropdown(
                            PROVIDERS["OpenAI"]["models"], value=PROVIDERS["OpenAI"]["default"],
                            label="Model", allow_custom_value=True,
                        )
                        api_key = gr.Textbox(
                            label="API key", type="password", placeholder="sk-...",
                            info="Sent directly to the selected provider for this "
                                 "session's requests only — we never store or log "
                                 "your key, so it's safe to use here.",
                        )
                        variety_type = gr.Radio(
                            ["Dialect", "ESL (L1 transfer)"], value="Dialect", label="Variety type"
                        )
                        dialect = gr.Dropdown(
                            list(DIALECTS), value=list(DIALECTS)[0], label="Dialect"
                        )
                        l1 = gr.Dropdown(
                            L1_LANGUAGES, value=L1_LANGUAGES[0], label="Native language (L1)",
                            visible=False,
                        )
                        cefr = gr.Dropdown(
                            CEFR_CHOICES, value=CEFR_CHOICES[0], label="CEFR level",
                            visible=False,
                            info=f"Applies {CEFR_SAMPLE_SIZE} randomly sampled CEFR "
                                 "simplification rules before the L1 errors (the full "
                                 "pipeline runs all of them offline).",
                        )
                        with gr.Accordion("Advanced", open=False):
                            shuffle_rules = gr.Checkbox(True, label="Shuffle rule order")
                            use_judge = gr.Checkbox(
                                True, label="Semantic-preservation judge",
                                info="Reject a rule's rewrite if the LLM judge finds meaning loss.",
                            )

                    with gr.Column(scale=2):
                        text = gr.Textbox(
                            label="Standard American English input", lines=4,
                            placeholder="Type or pick an example below…",
                        )
                        gr.Examples(EXAMPLES, inputs=text, label="Examples")
                        go = gr.Button("Transform", variant="primary")
                        status = gr.HTML()
                        result = gr.HTML()
                        with gr.Accordion("Rule-by-rule log", open=True):
                            log = gr.HTML()

            with gr.Tab("Examples gallery"):

                with gr.Row():
                    g_type = gr.Radio(
                        ["Dialect", "ESL (L1 × CEFR)"], value="Dialect",
                        label="Variety type",
                    )
                    g_variety = gr.Dropdown(
                        list(GALLERY_DIALECT_KEYS), value=list(GALLERY_DIALECT_KEYS)[0],
                        label="Variety",
                    )
                    _init_choices = gallery_choices("Dialect", list(GALLERY_DIALECT_KEYS)[0])
                    g_idx = gr.Dropdown(
                        _init_choices, value=_init_choices[0], label="Example",
                    )
                g_out = gr.HTML(
                    value=render_gallery(
                        "Dialect", list(GALLERY_DIALECT_KEYS)[0], _init_choices[0],
                    )
                )

        def on_gallery_type(t):
            options = list(GALLERY_DIALECT_KEYS) if t == "Dialect" else list(GALLERY_L1_KEYS)
            examples = gallery_choices(t, options[0])
            return (gr.update(choices=options, value=options[0]),
                    gr.update(choices=examples, value=examples[0]))

        def on_gallery_variety(t, v):
            examples = gallery_choices(t, v)
            return gr.update(choices=examples, value=examples[0])

        g_type.change(on_gallery_type, g_type, [g_variety, g_idx]).then(
            render_gallery, [g_type, g_variety, g_idx], g_out
        )
        g_variety.change(on_gallery_variety, [g_type, g_variety], g_idx).then(
            render_gallery, [g_type, g_variety, g_idx], g_out
        )
        g_idx.change(render_gallery, [g_type, g_variety, g_idx], g_out)

        def on_provider(p):
            return gr.update(
                choices=PROVIDERS[p]["models"], value=PROVIDERS[p]["default"]
            ), gr.update(placeholder=PROVIDERS[p]["key_hint"])

        provider.change(on_provider, provider, [model, api_key])

        def on_variety(v):
            is_dialect = v == "Dialect"
            return (gr.update(visible=is_dialect), gr.update(visible=not is_dialect),
                    gr.update(visible=not is_dialect))

        variety_type.change(on_variety, variety_type, [dialect, l1, cefr])

        go.click(
            transform,
            inputs=[text, variety_type, dialect, l1, cefr, provider, model, api_key,
                    shuffle_rules, use_judge],
            outputs=[result, log, status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_app().launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        theme=THEME, css=CSS,
    )
