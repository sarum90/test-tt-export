# CLAUDE.md

This file provides context for Claude Code when working on this project.

## Project Overview

Static site generator for linguistic text corpora with interlinear glossed text (IGT). Consumes a `passages.json` file exported from Twisted Tongues and generates a browsable, searchable static site.

## Tech Stack

- **Build**: Python + Jinja2 templates
- **Package management**: uv
- **Client-side search**: Alpine.js
- **Styling**: Plain CSS (no frameworks)
- **Deployment**: GitHub Pages via Actions

## Key Commands

```bash
uv sync                        # Install dependencies
uv run python build.py         # Build to public/
uv run python build.py --serve # Dev server with hot reload (port 8000)
```

## Architecture

### Data Structure

```
data/
├── passages.json       # Main data (from Twisted Tongues export)
├── about.md            # About page content
└── passages/           # Per-passage extras (auto-created on build)
    └── <slug>/
        ├── intro.md    # Optional intro (rendered above sentences)
        └── *.mp3       # Optional audio (auto-detected)
```

### Build Process (`build.py`)

1. Loads `data/passages.json` and strips internal Twisted Tongues links
2. Creates `data/passages/<slug>/` directories with empty `intro.md` for each passage
3. Removes orphaned passage directories (slugs that no longer exist)
4. Loads per-passage extras (intro.md content, audio files)
5. If ffmpeg available, converts audio to streaming (mp3) and lossless (flac) variants
6. Renders Markdown from `data/about.md`
7. Generates static HTML pages via Jinja2 templates
8. Copies audio variants to output alongside passage HTML
9. Copies `static/` to `public/`

### Audio Processing

With ffmpeg installed:
- Streaming: MP3 128kbps (for `<audio>` element)
- Lossless: FLAC (download link)
- Original: Preserved as-is (download link)

Cache in `data/audio_cache/<slug>/` - regenerated only when source mtime changes.

### Dev Server

- Watches `templates/`, `data/`, `static/` for changes
- 20ms debounce on rebuilds (catches save-alls)
- Uses absolute paths to survive `public/` deletion during rebuild

### Search (`templates/search.html`)

- Client-side search using Alpine.js
- Async with cooperative yielding (won't block UI)
- Cancellation support (new search aborts old one)
- Links directly to sentences via `#sentence-N` anchors

## Data Format

Passages contain sentences with:

- `word_tracks`: Aligned word-level data (IPA, gloss, etc.)
- `sentence_tracks`: Sentence-level translations
- `grammatical`: false = prefix with `*`
- `infelicitous`: true = prefix with `#`
- Superscripts: `^{content}` → `<sup>content</sup>`

## File Locations

| Purpose | Location |
|---------|----------|
| Passage data | `data/passages.json` |
| About content | `data/about.md` |
| Per-passage intro | `data/passages/<slug>/intro.md` |
| Per-passage audio | `data/passages/<slug>/*.mp3` (or .wav, .ogg, etc.) |
| Audio cache | `data/audio_cache/<slug>/` (gitignored) |
| All CSS | `templates/base.html` (inline `<style>`) |
| Search logic | `templates/search.html` (inline `<script>`) |

## Common Tasks

### Add a new page

1. Create template in `templates/`
2. Add rendering logic in `build.py` `build_site()`
3. Add nav link in `templates/base.html`

### Modify styles

All CSS is in `templates/base.html` in the `<style>` block. No external stylesheets.

### Change search behavior

Search JS is in `templates/search.html` in the `searchApp()` function.

### Add intro content to a passage

1. Run build once to create directories: `uv run python build.py`
2. Edit `data/passages/<slug>/intro.md`
3. Rebuild

### Add audio to a passage

1. Copy audio file to `data/passages/<slug>/`
2. Any `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac` file is auto-detected
3. Rebuild
