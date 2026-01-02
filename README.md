# Linguistic Passages Static Site

A static site generator for displaying linguistic texts with interlinear glossed text (IGT), built with Python and Jinja2.

## Features

- **Multi-page architecture**: Each passage gets its own URL for sharing and SEO
- **Statically rendered**: All content pre-rendered at build time
- **Search**: Client-side search across all passages with highlighted results
- **Deep linking**: Direct links to individual sentences (e.g., `/passages/story/#sentence-3`)
- **Per-passage extras**: Each passage can have intro text and audio
- **Responsive design**: Works on desktop and mobile
- **Hot reload**: Dev server auto-rebuilds on file changes
- **Simple tooling**: Just Python, no complex build chains

## Site Structure

```
/                                    → About page
/passages/                           → Passage list
/passages/elephant-story/            → Full passage with interlinear
/passages/elephant-story/#sentence-3 → Direct link to sentence
/search/                             → Search page (links to sentences)
```

## Quick Start

### 1. Install uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Replace the Data

Export your passages from Twisted Tongues and replace the sample data:

```bash
cp /path/to/your-export.json data/passages.json
```

### 4. Build Once to Create Passage Directories

```bash
uv run python build.py
```

This creates `data/passages/<slug>/` for each passage with stub `intro.md` files.

### 5. Customize Content

**About page**: Edit `data/about.md`

**Per-passage extras**: Each passage gets a directory in `data/passages/`:

```
data/passages/
├── the-elephant-story/
│   ├── intro.md          # Intro content (seeded from passages.json)
│   └── recording.mp3     # Optional audio (any audio file)
├── at-the-market/
│   └── intro.md
└── morning-greeting/
    └── intro.md
```

- **intro.md**: Markdown with YAML frontmatter. On first build, seeded with:
  ```markdown
  ---
  name: Passage Title
  description: Description from passages.json
  ---

  # {{ name }}

  {{ description }}
  ```
  The body uses Jinja2 syntax to reference frontmatter variables. This makes it explicit what's being rendered and lets you customize the layout (e.g., add content between title and description, use different heading levels, etc.).
- **Audio**: Any `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`, or `.aiff` file in the directory

### Audio Processing

If `ffmpeg` is installed, audio files are automatically converted:

- **Streaming**: MP3 128kbps for `<audio>` playback (small, fast loading)
- **Lossless**: FLAC for quality download
- **Original**: Raw file preserved for analysis

Converted files are cached in `data/audio_cache/` and only regenerated when the source changes.

Without ffmpeg, the original file is used directly (no conversion).

### 6. Build and Serve

```bash
# Development server with hot reload
uv run python build.py --serve

# Build for production
uv run python build.py
```

The built site will be in `public/`.

## Project Structure

```
├── pyproject.toml          # Python project configuration
├── build.py                # Build script with dev server
├── data/
│   ├── passages.json       # Your passage data (replace this!)
│   ├── about.md            # About page content
│   └── passages/           # Per-passage extras (auto-created)
│       └── <slug>/
│           ├── intro.md    # Optional intro content
│           └── *.mp3       # Optional audio file
├── static/                 # Static assets (copied to output)
└── templates/
    ├── base.html           # Base layout, navigation, all CSS
    ├── about.html          # About page
    ├── passages.html       # Passage list
    ├── passage.html        # Individual passage with interlinear
    └── search.html         # Search page with Alpine.js
```

## Data Format

The `passages.json` file follows the Twisted Tongues export format:

```json
{
  "passages": [
    {
      "name": "Passage Title",
      "description": "Optional description",
      "sentences": [
        {
          "grammatical": true,
          "infelicitous": false,
          "word_tracks": [
            {"name": "IPA", "words": ["ɔ^{3}", "li^{3}"]},
            {"name": "Gloss", "words": ["3SG", "eat.PFV"]}
          ],
          "sentence_tracks": [
            {"name": "English", "sentence": "He ate."}
          ]
        }
      ]
    }
  ]
}
```

### Special Formatting

- **Superscripts**: Use `^{content}` for tone marks (e.g., `e^{4}` → e⁴)
- **IPA Characters**: Use Unicode directly (e.g., `ɔ`, `ŋ`, `ʃ`)
- **Grammaticality**: `"grammatical": false` shows `*` prefix
- **Infelicitous**: `"infelicitous": true` shows `#` prefix

## Updating Passages

When you re-export `passages.json` from Twisted Tongues:

1. Replace `data/passages.json`
2. Run `uv run python build.py`
3. New passages get directories with `intro.md` seeded from passages.json
4. Existing `intro.md` files are preserved — your custom names, descriptions, and content are kept
5. Orphaned directories (passages no longer in JSON) are automatically removed

**Note**: The `name` and `description` in intro.md frontmatter take precedence over passages.json. This lets you customize display names and descriptions independently of the source data.

## Deployment

### GitHub Pages (Recommended)

This repo includes a GitHub Action that automatically builds and deploys on push to `main`.

1. Go to your repo's **Settings → Pages**
2. Set **Source** to "GitHub Actions"
3. Push to `main` — the site deploys automatically

### Other Hosts

```bash
uv run python build.py --base-url "https://your-domain.com/"
```

For subdirectory deployment (e.g., `https://user.github.io/repo/`):

```bash
uv run python build.py --base-url "https://user.github.io/repo/"
```

## Credits

- Interactivity: [Alpine.js](https://alpinejs.dev/)
- Data format: [Twisted Tongues](https://twisted-tongues-beta.appspot.com/)
