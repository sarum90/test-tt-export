# Linguistic Passages Static Site

A static site for displaying linguistic texts with interlinear glossed text (IGT), built with [Zola](https://www.getzola.org/) and [Alpine.js](https://alpinejs.dev/).

## Features

- **Multi-page architecture**: Each passage gets its own URL for sharing and SEO
- **Statically rendered**: All content pre-rendered at build time
- **Search**: Client-side search across all passages with highlighted results
- **Audio support**: Optional per-passage audio playback
- **Responsive design**: Works on desktop and mobile
- **Minimal JavaScript**: Only used for search and superscript rendering

## Architecture

```
/                           → About page
/passages/                  → Passage list (cards)
/passages/elephant-story/   → Full passage with interlinear
/passages/market/           → Full passage with interlinear
/search/                    → Search page (links to passages)
```

Each passage is a separate HTML page, making the site fast and scalable.

## Quick Start

### 1. Install Zola

See [Zola installation guide](https://www.getzola.org/documentation/getting-started/installation/).

```bash
# macOS
brew install zola

# Ubuntu/Debian
sudo snap install zola

# Arch
sudo pacman -S zola
```

### 2. Replace the Data

Export your passages from Twisted Tongues and replace the sample data:

```bash
cp /path/to/your-export.json data/passages.json
```

### 3. Generate Passage Pages

Run the generator script to create individual passage pages:

```bash
python scripts/generate_passages.py --clean
```

This creates a markdown file in `content/passages/` for each passage.

### 4. Customize the About Page

Edit `content/_index.md` with your project description, contributors, and citation info.

### 5. Add Audio (Optional)

Create or edit `data/audio.json` to map passage names to audio files:

```json
{
  "The Elephant Story": "audio/elephant.mp3",
  "At the Market": null,
  "Morning Greeting": "audio/greetings.mp3"
}
```

Then add your audio files to `static/audio/`.

### 6. Build and Serve

```bash
# Development server with hot reload
zola serve

# Build for production
zola build
```

The built site will be in `public/`.

## Workflow Summary

After updating `passages.json`:

```bash
python scripts/generate_passages.py --clean   # Regenerate passage pages
python scripts/generate_audio_manifest.py     # Update audio manifest (optional)
zola build                                    # Build the site
```

## Project Structure

```
static_tt/
├── config.toml                 # Site configuration
├── content/
│   ├── _index.md              # About page (edit this!)
│   ├── search.md              # Search page config
│   └── passages/
│       ├── _index.md          # Passages list config
│       └── *.md               # Generated passage pages
├── data/
│   ├── passages.json          # Your passage data (replace this!)
│   └── audio.json             # Audio file mapping
├── static/
│   ├── audio/                 # Audio files
│   └── js/app.js              # Minimal JavaScript
├── templates/
│   ├── base.html              # Base layout with navigation
│   ├── about.html             # About page template
│   ├── passages.html          # Passage list template
│   ├── passage.html           # Individual passage template
│   ├── search.html            # Search page template
│   └── macros.html            # Interlinear rendering macros
├── sass/style.scss            # Custom styles
├── scripts/
│   ├── generate_passages.py   # Generate passage pages from JSON
│   └── generate_audio_manifest.py  # Generate audio.json template
└── README.md
```

## Data Format

The `passages.json` file follows the Twisted Tongues export format:

```json
{
  "passages": [
    {
      "name": "Passage Title",
      "description": "Optional description",
      "link": "https://source-url",
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
          ],
          "link": "https://sentence-url"
        }
      ]
    }
  ]
}
```

### Special Formatting

- **Superscripts**: Use `^{content}` for tone marks (e.g., `e^{4}` renders as e⁴)
- **IPA Characters**: Use Unicode directly (e.g., `ɔ`, `ŋ`, `ʃ`)
- **Grammaticality**: `"grammatical": false` adds `*` prefix
- **Infelicitous**: `"infelicitous": true` adds `#` prefix

## Deployment

### GitHub Pages (Recommended)

This repo includes a GitHub Action that automatically builds and deploys on push to `main`.

1. Go to your repo's **Settings → Pages**
2. Set **Source** to "GitHub Actions"
3. Push to `main` — the site deploys automatically

The action runs `generate_passages.py` and `zola build` for you.

### Other Hosts

- **Netlify**: Set build command to `python scripts/generate_passages.py && zola build`
- **Vercel**: Similar to Netlify
- **Manual**: Run the build locally and upload `public/`

## Credits

- Static site generation: [Zola](https://www.getzola.org/)
- Interactivity: [Alpine.js](https://alpinejs.dev/)
- Styling: [Tailwind CSS](https://tailwindcss.com/)
- Data format: [Twisted Tongues](https://twisted-tongues-beta.appspot.com/)
