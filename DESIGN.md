---
version: alpha
name: AeroPulse-demo
description: Design contract for the AeroPulse Streamlit demo (turbofan Remaining Useful Life on NASA C-MAPSS). Utilitarian, document-style minimalism, adapted from the taste-skill minimalist-ui protocol to what Streamlit can render. The surface is a warm-monochrome canvas with near-black type. Colour appears only as muted semantic pastels (late, early, healthy, capped), and the charts are the only visual argument.

colors:
  canvas: "#FFFFFF"
  canvas-warm: "#F7F6F3"
  surface: "#FBFBFA"
  ink: "#2F3437"
  ink-strong: "#111111"
  ink-mute: "#787774"
  ink-faint: "#B4B4B0"
  hairline: "#EAEAEA"
  action: "#111111"
  action-hover: "#333333"
  late-bg: "#FDEBEC"
  late-ink: "#9F2F2D"
  early-bg: "#E1F3FE"
  early-ink: "#1F6C9F"
  healthy-bg: "#EDF3EC"
  healthy-ink: "#346538"
  capped-bg: "#FBF3DB"
  capped-ink: "#956400"

typography:
  title:
    fontFamily: "Newsreader, 'Instrument Serif', Georgia, serif"
    fontSize: 36px
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: -0.03em
  heading:
    fontFamily: "Geist, 'Helvetica Neue', Arial, sans-serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: -0.01em
  body:
    fontFamily: "Geist, 'Helvetica Neue', Arial, sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0
  caption:
    fontFamily: "Geist, 'Helvetica Neue', Arial, sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  badge:
    fontFamily: "Geist, 'Helvetica Neue', Arial, sans-serif"
    fontSize: 11px
    fontWeight: 600
    lineHeight: 1.0
    letterSpacing: 0.05em
  metric:
    fontFamily: "'Geist Mono', 'SF Mono', 'JetBrains Mono', ui-monospace, monospace"
    fontSize: 32px
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: -0.02em
  code:
    fontFamily: "'Geist Mono', 'SF Mono', 'JetBrains Mono', ui-monospace, monospace"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0

rounded:
  sm: 4px
  md: 6px
  lg: 8px
  full: 9999px

spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 40px

components:
  metric-tile:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.metric}"
    rounded: "{rounded.lg}"
    padding: 24px
  status-badge-healthy:
    backgroundColor: "{colors.healthy-bg}"
    textColor: "{colors.healthy-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.full}"
    padding: 2px 8px
  status-badge-alert:
    backgroundColor: "{colors.late-bg}"
    textColor: "{colors.late-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.full}"
    padding: 2px 8px
  status-badge-capped:
    backgroundColor: "{colors.capped-bg}"
    textColor: "{colors.capped-ink}"
    typography: "{typography.badge}"
    rounded: "{rounded.full}"
    padding: 2px 8px
  note-disclaimer:
    backgroundColor: "{colors.canvas-warm}"
    textColor: "{colors.ink-mute}"
    typography: "{typography.caption}"
    rounded: "{rounded.md}"
    padding: 12px 16px
  chart-panel:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.caption}"
    rounded: "{rounded.lg}"
    padding: 16px
  sidebar:
    backgroundColor: "{colors.canvas-warm}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: 24px
---

## Overview

The demo is a working instrument, not a landing page. A visitor picks a subset and an engine and reads three numbers, a curve and an explanation. The design's job is to make those readable and honest, and to stay out of the way.

- Canvas `{colors.canvas}` white. The sidebar sits on `{colors.canvas-warm}` bone. Body text is `{colors.ink}` charcoal, never pure black.
- One action colour, `{colors.action}` near-black, for sliders, focus rings and the few buttons. There is no brand hue.
- Colour carries meaning only, through four muted pastel pairs:
  - late (`late-*`)
  - early (`early-*`)
  - healthy (`healthy-*`)
  - capped / warning (`capped-*`)
- Every container edge is a 1px `{colors.hairline}` line. Shadows are effectively absent.

Source protocol: taste-skill `minimalist-ui` (Premium Utilitarian Minimalism). The adaptations to Streamlit are listed under "Deviations from the protocol".

## Colors

| Token | Hex | Use |
|---|---|---|
| `canvas` | `#FFFFFF` | Main area background, chart backgrounds |
| `canvas-warm` | `#F7F6F3` | Sidebar (`secondaryBackgroundColor`), disclaimer note, warm-up band on the curve |
| `surface` | `#FBFBFA` | Expander bodies, code blocks |
| `ink` | `#2F3437` | Body text, chart text, predicted-RUL line |
| `ink-strong` | `#111111` | Title, metric values |
| `ink-mute` | `#787774` | Captions, axis ticks, helper copy, true-RUL dashed line |
| `ink-faint` | `#B4B4B0` | Disabled text, chart minor elements |
| `hairline` | `#EAEAEA` | Borders, dividers, chart grid |
| `action` | `#111111` | Streamlit `primaryColor`: slider thumb/track, focus, selected widget state |

### Semantic pastels (the only chromatic colours)

| Meaning | Background | Ink | Used for |
|---|---|---|---|
| Late / alert | `#FDEBEC` | `#9F2F2D` | "Schedule maintenance" badge, late errors in the worst-case list, SHAP contributions that lower predicted RUL, the alert-threshold line, the late branch of the NASA cost curve |
| Early | `#E1F3FE` | `#1F6C9F` | Early errors in the worst-case list, SHAP contributions that raise predicted RUL, the early branch of the NASA cost curve |
| Healthy | `#EDF3EC` | `#346538` | "Healthy" badge |
| Capped / caution | `#FBF3DB` | `#956400` | "125+ (model capped)" badge, the warm-up note, "above the 125 training cap" note |

Rule: a pastel background always pairs with its own ink. Saturated reds, greens, blues or gradients never appear, including the SHAP library's default `#ff0051` / `#008bfb`, which must be overridden.

## Typography

- **Title** (one per page, the app title only): Newsreader serif, 500, tight tracking. Falls back to Georgia.
- **UI and body:** Geist sans. Falls back to Helvetica Neue / Arial.
- **Numbers:** Geist Mono with tabular figures for metric values, signed errors (`+41`, `-12`) and code.
- Banned: Inter, Roboto, Open Sans.
- Line height 1.6 for body text. Captions sit at `{typography.caption}` 13px in `{colors.ink-mute}`.
- Fonts are configured through Streamlit's theme (`font`, `headingFont`, `codeFont`, and `fontFaces` if the pinned Streamlit supports them). If a webfont can't load (offline, a Cloud restriction), the fallback chain must still look correct. Never block rendering on a font.
- Matplotlib figures can't rely on webfonts. They use `'Helvetica Neue', 'Arial', 'DejaVu Sans'` (DejaVu is what Cloud's Linux image has) at 9-10pt, `{colors.ink}` text and `{colors.ink-mute}` ticks.

## Layout (Streamlit mapping)

Page order, top to bottom. The structure is fixed by `03-CONTEXT.md` D-04; this file sets only its look.

1. **Title + one-line subtitle.** Serif title, mute caption: "XGBoost on causal rolling-window features, NASA C-MAPSS FD001-FD004".
2. **Disclaimer note** (`note-disclaimer`): bone background, mute 13px text, no icon or a single Material icon. Always visible.
3. **Metrics row:** three `metric-tile`s (`st.container(border=True)` with `st.metric`, or the equivalent) for Predicted RUL, True RUL and Status. The status shows as a pill badge (`status-badge-*`) under or beside the value. Values use the mono `metric` style.
4. **Two chart panels** side by side (`chart-panel`, 1px hairline, 8px radius): predicted-vs-true curve on the left, SHAP waterfall on the right. On narrow screens they stack, which is Streamlit's default.
5. **Expanders** (accordion style: hairline separators, no heavy boxes): "Why late predictions cost more", "Sensor trends", "How to read this".

Spacing:
- Use Streamlit's default block gap, plus `{spacing.lg}` around the metrics row.
- Keep the main area at the default centered width or `layout="wide"`, whichever keeps the two chart panels legible. Do not exceed about 1200px of content width.

Sidebar, top to bottom: subset, engine, worst-case picker, alert threshold, "Observed up to cycle". Labels are plain sentences in sentence case.

## Charts (matplotlib)

One shared style helper (for example `demo_lib.plot_style()` returning an rc dict, or a context manager) applies to every figure:
- `figure.facecolor` / `axes.facecolor` `#FFFFFF`; top and right spines off; left and bottom spines `#EAEAEA`.
- Grid: y only, `#EAEAEA`, 0.8 px. Ticks `#787774`. Labels `#2F3437`. No chart titles inside figures; the panel caption names the chart.
- Curve:
  - predicted RUL: solid `#2F3437`, 2 px
  - true RUL (capped, reconstructed): dashed `#787774`, 1.5 px
  - alert threshold: dotted `#9F2F2D`, 1 px
  - cycles 1-9: an `axvspan` band in `#F7F6F3`
  - the current cycle: a small `#111111` marker
- SHAP waterfall: override shap's default colours so contributions that raise RUL use `#1F6C9F` and those that lower it use `#9F2F2D`. Keep the top 10 + "other N features" row. Size the figure to match the curve panel.
- NASA cost curve: early branch `#1F6C9F`, late branch `#9F2F2D`, the engine's marker in `#111111` with its value annotated in mono.
- Legends: `frameon=False`, 8pt, `#787774`.
- Always `plt.close(fig)` after `st.pyplot(fig)`.
- Streamlit-native charts (sensor trends) take the muted categorical palette below through theme config where supported: `#2F3437, #1F6C9F, #9F2F2D, #346538, #956400, #787774`.

## Components

- **Status badge:** a pill with uppercase micro label and wide tracking, pastel by state: `HEALTHY`, `SCHEDULE MAINTENANCE`, `125+ CAPPED`. Use `st.badge` if the pinned Streamlit has it and its colours can be themed to the pastels. Otherwise use a static-string HTML span. Interpolate only numbers or fixed labels, never data strings.
- **Worst-case list entries:** `Engine 34 · +41 late` style. The signed number is mono, and the late/early word is tinted with its pastel ink.
- **Notes:**
  - The warm-up note uses the `capped` pastel pair.
  - The "above the 125 training cap" note uses the same pair, at caption size.
  - Avoid stacking multiple `st.info` / `st.warning` boxes. Prefer one quiet note.
- **Buttons:** if any exist, `action` fill, white text, 6px radius, no shadow.

## Copy rules

- Plain, specific language. Name the quantity and the unit ("cycles").
- No emojis anywhere, including `page_icon`. Use no icon, or a Material icon (`:material/...:`).
- No marketing clichés: elevate, seamless, unleash, next-gen, game-changer, delve.
- No safety or business claims. The data is simulated and this is a demo.
- Every number from evaluation (the model card) is read from `outputs/results_baseline.json` at runtime, never typed.

## Configuration

- `.streamlit/config.toml` (committed; Cloud reads it) holds the `[theme]` block:
  - `base = "light"`
  - `primaryColor = "#111111"`
  - `backgroundColor = "#FFFFFF"`
  - `secondaryBackgroundColor = "#F7F6F3"`
  - `textColor = "#2F3437"`
  - plus whatever the pinned Streamlit 1.65 supports among `font`, `headingFont`, `codeFont`, `fontFaces`, `baseRadius`, `borderColor`, `showWidgetBorder`, `chartCategoricalColors` and semantic colour keys

  Research confirms the exact keys against Streamlit 1.65. Unknown keys must not be added.
- Keep custom CSS to zero if possible. If a small `<style>` block is unavoidable (for example tabular numerals), it holds static CSS only.
- Never commit `.streamlit/secrets.toml`; it's already gitignored.

## Deviations from the protocol (deliberate)

| minimalist-ui says | Demo does | Why |
|---|---|---|
| Scroll-entry, staggered reveals, ambient gradient blobs | None | Streamlit re-renders on every interaction, so animations would replay on each slider move. Motion would also distract from the data. |
| Hero imagery, photos, illustrations, background textures | None | A data demo: the curve and waterfall are the visual content. Images add weight and cold-start time. |
| Bento grids, faux-OS window chrome, `<kbd>` keys | Not used | No marketing surface and no keyboard shortcuts. |
| Phosphor / Radix icon sets | Streamlit Material icons only, sparingly | Streamlit natively supports `:material/...:`. Adding an icon library would mean custom HTML. |
| Massive section padding (`py-24`+) | Streamlit's default gaps | A single dense screen is the right shape for an instrument. |
| Metric values in the mono `metric` style (Geist Mono) | Metric values use the body font Geist | No theme key sets a metric font family, and the demo uses no custom CSS. `codeFont` does not apply to `st.metric`. |
| Fonts always render as designed | Offline or when a webfont is blocked, the fonts fall back to Streamlit's bundled Source Sans and Source Code Pro, so the title loses its serif | A theme font entry that carries a URL cannot also carry a fallback chain: a comma list after the URL is swallowed into the URL. |
| The warm-up note uses a quiet capped-pair note | The warm-up note is Streamlit's info alert (kept by decision D-10) carrying a yellow WARM-UP badge for the capped pair | `st.info` fills cannot be themed without CSS, so the capped pastel reaches the note only through the badge. |
| The disclaimer note has a bone background | The disclaimer is a hairline-bordered container with a mute caption | Container fills are not themeable without CSS; the border and the mute caption carry the look. |
| Worst-case entries tint the late and early word and set the number in mono | Worst-case entries are plain text in a selectbox | The options of a selectbox cannot be tinted or styled without CSS. The above-cap tag is part of the label text. |
| Chart text follows the 9-10 pt rule | shap draws the waterfall tick labels at a fixed 13 pt | The size is fixed inside shap's plotting code and no style key reaches it. |

## Do and don't

**Do**
- Keep colour semantic, so a visitor learns that red means late and blue means early.
- Use one quiet disclaimer and one quiet note at a time.
- Close every matplotlib figure.

**Don't**
- Use saturated or default library colours.
- Use emojis.
- Hard-code evaluation numbers.
- Use dark-mode-only assets.
- Use gradients.
- Use shadows heavier than `0 1px 2px rgba(0,0,0,0.04)`.
