#!/usr/bin/env python3
"""
Build script for linguistic passages static site.

Renders JSON passage data to static HTML pages using Jinja2 templates.

Usage:
    python build.py              # Build to public/
    python build.py --serve      # Build and serve locally
    python build.py --clean      # Clean public/ before building
"""

import argparse
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path

import frontmatter
from jinja2 import Environment, FileSystemLoader
import markdown
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# --- Configuration ---

OUTPUT_DIR = Path("public")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
DATA_DIR = Path("data")
PASSAGES_DATA_DIR = DATA_DIR / "passages"

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aiff', '.aif'}
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
IMAGE_CACHE_DIR = DATA_DIR / "image_cache"
IMAGE_WIDTHS = [600, 1200]  # 1x and 2x for ~600px prose container


# --- Utility Functions ---

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    slug = text.lower()
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-') or 'untitled'


def strip_links(data: dict) -> dict:
    """Remove internal Twisted Tongues links from passages data."""
    cleaned = deepcopy(data)
    for passage in cleaned.get('passages', []):
        passage.pop('link', None)
        for sentence in passage.get('sentences', []):
            sentence.pop('link', None)
    return cleaned


def render_superscripts(text: str) -> str:
    """Convert ^{...} notation to <sup>...</sup> HTML."""
    if not text:
        return ''
    return re.sub(r'\^\{([^}]*)\}', r'<sup>\1</sup>', text)


def get_available_tracks(passage: dict, display_order: list = None) -> list:
    """Get list of track names available in this passage, in display order."""
    word_track_names = set()
    sentence_track_names = set()

    for sentence in passage.get('sentences', []):
        for track in sentence.get('word_tracks', []):
            word_track_names.add(track.get('name'))
        for track in sentence.get('sentence_tracks', []):
            sentence_track_names.add(track.get('name'))

    all_tracks = word_track_names | sentence_track_names

    # Return in config-specified order, then any remaining tracks alphabetically
    if display_order:
        ordered = [name for name in display_order if name in all_tracks]
        remaining = sorted(all_tracks - set(ordered))
        return ordered + remaining
    return sorted(all_tracks)


def get_valid_sentences(passage: dict) -> list:
    """Filter out empty sentences from a passage.

    A sentence is valid if it has either:
    - word_tracks with at least one word, OR
    - sentence_tracks with content
    """
    sentences = []
    for s in passage.get('sentences', []):
        word_tracks = s.get('word_tracks', [])
        sentence_tracks = s.get('sentence_tracks', [])

        has_words = word_tracks and word_tracks[0].get('words') and len(word_tracks[0]['words']) > 0
        has_translations = sentence_tracks and any(t.get('sentence') for t in sentence_tracks)

        if has_words or has_translations:
            sentences.append(s)
    return sentences


def render_markdown(text: str) -> str:
    """Render markdown text to HTML."""
    return markdown.markdown(text, extensions=['extra'])


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_imagemagick() -> bool:
    """Check if ImageMagick convert is available."""
    try:
        subprocess.run(['convert', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def process_image(source_file: Path, slug: str) -> dict:
    """
    Process image into responsive variants.

    Returns dict with paths to generated variants at different widths,
    plus WebP versions for modern browsers.
    """
    source_file = source_file.resolve()
    cache_dir = IMAGE_CACHE_DIR.resolve() / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_stem = source_file.stem
    source_ext = source_file.suffix.lower()
    source_mtime = source_file.stat().st_mtime

    variants = {
        'original': source_file,
        'widths': {},
    }

    # Get original dimensions
    result = subprocess.run(
        ['identify', '-format', '%w', str(source_file)],
        capture_output=True, text=True
    )
    original_width = int(result.stdout.strip()) if result.returncode == 0 else 9999

    for width in IMAGE_WIDTHS:
        # Skip if original is smaller than target
        if width > original_width:
            continue

        # Generate resized JPEG/PNG
        resized_path = cache_dir / f"{source_stem}-{width}w{source_ext}"
        webp_path = cache_dir / f"{source_stem}-{width}w.webp"

        def needs_regen(cached: Path) -> bool:
            if not cached.exists():
                return True
            return source_mtime > cached.stat().st_mtime

        if needs_regen(resized_path):
            print(f"  Resizing to {width}w: {source_file.name}")
            subprocess.run([
                'convert', str(source_file),
                '-resize', f'{width}x>',
                '-quality', '85',
                str(resized_path)
            ], capture_output=True, check=True)

        if needs_regen(webp_path):
            print(f"  Converting to WebP {width}w: {source_file.name}")
            subprocess.run([
                'convert', str(source_file),
                '-resize', f'{width}x>',
                '-quality', '85',
                str(webp_path)
            ], capture_output=True, check=True)

        variants['widths'][width] = {
            'fallback': resized_path,
            'webp': webp_path,
        }

    # Also generate a WebP of the original size (capped at largest breakpoint)
    if original_width > IMAGE_WIDTHS[-1]:
        max_width = IMAGE_WIDTHS[-1]
    else:
        max_width = original_width

    return variants


class IntroRenderer:
    """Renders intro.md content with media helpers for Jinja2 templates."""

    def __init__(self, slug: str, passage_dir: Path):
        self.slug = slug
        self.passage_dir = passage_dir
        self.files_to_copy = []  # List of (source_path, dest_name) tuples
        self.audio_data = None   # Audio URLs for template

    def image(self, src: str, alt: str = "", caption: str = None) -> str:
        """
        Generate responsive image HTML.

        Usage in intro.md:
            {{ image("photo.jpg", alt="Description", caption="Photo credit") }}
        """
        # Skip external URLs
        if src.startswith(('http://', 'https://', '//')):
            img = f'<img src="{src}" alt="{alt}" loading="lazy">'
            if caption:
                return f'<figure>{img}<figcaption>{caption}</figcaption></figure>'
            return img

        source_file = self.passage_dir / src
        if not source_file.exists():
            return f'<!-- image not found: {src} -->'

        # SVGs don't need processing
        if source_file.suffix.lower() == '.svg':
            self.files_to_copy.append((source_file, src))
            img = f'<img src="{src}" alt="{alt}" loading="lazy">'
            if caption:
                return f'<figure>{img}<figcaption>{caption}</figcaption></figure>'
            return img

        # Process image into responsive variants
        variants = process_image(source_file, self.slug)

        # Build srcset strings
        webp_srcset = []
        fallback_srcset = []

        sorted_widths = sorted(variants['widths'].keys())
        for width in sorted_widths:
            v = variants['widths'][width]
            webp_srcset.append(f"{v['webp'].name} {width}w")
            fallback_srcset.append(f"{v['fallback'].name} {width}w")
            self.files_to_copy.append((v['webp'], v['webp'].name))
            self.files_to_copy.append((v['fallback'], v['fallback'].name))

        if not sorted_widths:
            # Image smaller than smallest breakpoint, copy original
            self.files_to_copy.append((source_file, src))
            img = f'<img src="{src}" alt="{alt}" loading="lazy">'
            if caption:
                return f'<figure>{img}<figcaption>{caption}</figcaption></figure>'
            return img

        # Build picture element
        largest_width = sorted_widths[-1]
        largest_fallback = variants['widths'][largest_width]['fallback'].name

        picture = f'''<picture>
  <source type="image/webp" srcset="{', '.join(webp_srcset)}" sizes="(max-width: 640px) 100vw, 600px">
  <img src="{largest_fallback}" srcset="{', '.join(fallback_srcset)}" sizes="(max-width: 640px) 100vw, 600px" alt="{alt}" loading="lazy">
</picture>'''

        if caption:
            return f'<figure>{picture}<figcaption>{caption}</figcaption></figure>'
        return picture

    def audio(self, src: str, title: str = None) -> str:
        """
        Generate audio player HTML with download links.

        Usage in intro.md:
            {{ audio("recording.wav") }}
            {{ audio("recording.wav", title="Original recording") }}
        """
        source_file = self.passage_dir / src
        if not source_file.exists():
            return f'<!-- audio not found: {src} -->'

        # Process audio into variants
        variants = get_audio_variants(source_file, self.slug)

        def format_size(path: Path) -> str:
            size = path.stat().st_size
            if size >= 1024 * 1024:
                return f"{size / (1024 * 1024):.1f} MB"
            elif size >= 1024:
                return f"{size / 1024:.0f} KB"
            return f"{size} B"

        # Track files to copy and build URLs
        streaming_name = variants['streaming'].name
        self.files_to_copy.append((variants['streaming'], streaming_name))

        downloads = [f'<a href="{streaming_name}" download>MP3 <span class="text-muted">({format_size(variants["streaming"])})</span></a>']

        if 'lossless' in variants and variants['lossless'] != variants['streaming']:
            lossless_name = variants['lossless'].name
            self.files_to_copy.append((variants['lossless'], lossless_name))
            downloads.append(f'<a href="{lossless_name}" download>FLAC <span class="text-muted">(lossless, {format_size(variants["lossless"])})</span></a>')

        if variants['original'] not in (variants['streaming'], variants.get('lossless')):
            original_name = variants['original'].name
            self.files_to_copy.append((variants['original'], original_name))
            downloads.append(f'<a href="{original_name}" download>Original <span class="text-muted">({format_size(variants["original"])})</span></a>')

        title_html = f'<div class="audio-title">{title}</div>' if title else ''

        return f'''<div class="audio-container card card-padding">
  {title_html}
  <audio controls preload="metadata">
    <source src="{streaming_name}">
    Your browser does not support audio playback.
  </audio>
  <div class="audio-downloads mt-2">
    <span class="text-sm text-muted">Download:</span>
    {" ".join(downloads)}
  </div>
</div>'''


def get_audio_variants(source_file: Path, slug: str) -> dict:
    """
    Process audio file into variants for web delivery.

    Returns dict with paths to:
    - streaming: lossy mp3 for <audio> playback (~128kbps)
    - lossless: flac for quality download
    - original: original file as-is
    """
    # Use absolute paths to avoid cwd issues
    source_file = source_file.resolve()
    cache_dir = AUDIO_CACHE_DIR.resolve() / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_stem = source_file.stem
    source_ext = source_file.suffix.lower()
    source_mtime = source_file.stat().st_mtime

    # Define output paths
    streaming_path = cache_dir / f"{source_stem}.mp3"
    lossless_path = cache_dir / f"{source_stem}.flac"

    variants = {
        'original': source_file,
        'original_name': source_file.name,
    }

    # Check if we need to regenerate (source newer than cached)
    def needs_regen(cached: Path) -> bool:
        if not cached.exists():
            return True
        return source_mtime > cached.stat().st_mtime

    # Generate streaming version (mp3 128kbps)
    if source_ext == '.mp3':
        # Already mp3, use as-is for streaming
        variants['streaming'] = source_file
    elif needs_regen(streaming_path):
        print(f"  Converting to mp3: {source_file.name}")
        subprocess.run([
            'ffmpeg', '-y', '-i', str(source_file),
            '-codec:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '44100', '-ac', '2',
            str(streaming_path)
        ], capture_output=True, check=True)
        variants['streaming'] = streaming_path
    else:
        variants['streaming'] = streaming_path

    # Generate lossless version (flac)
    if source_ext == '.flac':
        # Already flac, use as-is
        variants['lossless'] = source_file
    elif needs_regen(lossless_path):
        print(f"  Converting to flac: {source_file.name}")
        subprocess.run([
            'ffmpeg', '-y', '-i', str(source_file),
            '-codec:a', 'flac',
            str(lossless_path)
        ], capture_output=True, check=True)
        variants['lossless'] = lossless_path
    else:
        variants['lossless'] = lossless_path

    return variants


# --- Configuration ---

VERSION_DEFAULTS = {
    "V1": {"word_tracks": [], "sentence_tracks": []},
    "V2": {"word_tracks": [], "sentence_tracks": ["English", "French"]},
    "V3": {"word_tracks": ["IPA"], "sentence_tracks": ["English", "French"]},
    "V4": {"word_tracks": ["IPA", "Gloss"], "sentence_tracks": ["English", "French"]},
}


def load_config() -> dict:
    """Load site configuration from data/config.json."""
    config_path = DATA_DIR / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "version": "V4",
        "site_title": "Language Passages",
        "site_description": "Interactive linguistic text corpus with interlinear glosses",
        "version_definitions": VERSION_DEFAULTS,
    }


def filter_tracks(passages_data: dict, config: dict) -> dict:
    """Filter word_tracks and sentence_tracks based on version config."""
    version = config.get('version', 'V4')
    version_defs = config.get('version_definitions', VERSION_DEFAULTS)
    version_config = version_defs.get(version, VERSION_DEFAULTS['V4'])

    allowed_word_tracks = set(version_config.get('word_tracks', []))
    allowed_sentence_tracks = set(version_config.get('sentence_tracks', []))

    filtered = deepcopy(passages_data)
    for passage in filtered.get('passages', []):
        for sentence in passage.get('sentences', []):
            # Filter word_tracks
            sentence['word_tracks'] = [
                t for t in sentence.get('word_tracks', [])
                if t.get('name') in allowed_word_tracks
            ]
            # Filter sentence_tracks
            sentence['sentence_tracks'] = [
                t for t in sentence.get('sentence_tracks', [])
                if t.get('name') in allowed_sentence_tracks
            ]
    return filtered


def load_glossary_content(lang: str = 'en') -> str:
    """Load and render glossary.md content for specified language."""
    # Try language-specific file first, fall back to default
    if lang != 'en':
        glossary_path = DATA_DIR / f"glossary-{lang}.md"
        if glossary_path.exists():
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return render_markdown(f.read())
    glossary_path = DATA_DIR / "glossary.md"
    if glossary_path.exists():
        with open(glossary_path, 'r', encoding='utf-8') as f:
            return render_markdown(f.read())
    return ""


def load_i18n() -> dict:
    """Load internationalization strings."""
    i18n_path = DATA_DIR / "i18n.json"
    if i18n_path.exists():
        with open(i18n_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"en": {}, "fr": {}}


# --- Build Functions ---

def load_data() -> dict:
    """Load passages data from JSON file."""
    passages_path = DATA_DIR / "passages.json"

    with open(passages_path, 'r', encoding='utf-8') as f:
        passages_data = json.load(f)

    # Strip internal links
    passages_data = strip_links(passages_data)

    return passages_data


def ensure_passage_dirs(passages: list) -> None:
    """Create passage data directories and seed intro.md files. Clean up orphans."""
    PASSAGES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get all valid slugs
    valid_slugs = set()
    for p in passages:
        slug = slugify(p.get('name', 'untitled'))
        valid_slugs.add(slug)
        passage_dir = PASSAGES_DATA_DIR / slug
        passage_dir.mkdir(exist_ok=True)

        # Seed intro.md with frontmatter if empty or missing
        intro_path = passage_dir / "intro.md"
        needs_seed = not intro_path.exists() or intro_path.stat().st_size == 0
        if needs_seed:
            name = p.get('name', 'Untitled')
            description = p.get('description', '')
            # Body uses Jinja2 helpers for frontmatter and media
            body = '''# {{ name }}

{{ description }}

{# Add media with: {{ image("photo.jpg", alt="...", caption="...") }} #}
{# Add audio with: {{ audio("recording.wav", title="...") }} #}
'''
            post = frontmatter.Post(content=body, name=name, description=description)
            intro_path.write_text(frontmatter.dumps(post), encoding='utf-8')

    # Remove orphaned directories (slugs that no longer exist)
    for item in PASSAGES_DATA_DIR.iterdir():
        if item.is_dir() and item.name not in valid_slugs:
            shutil.rmtree(item)
            print(f"Removed orphaned: data/passages/{item.name}/")


def load_passage_extras(slug: str, lang: str = 'en') -> dict:
    """Load extra data for a passage from its data directory."""
    passage_dir = PASSAGES_DATA_DIR / slug
    extras = {
        'name': None,
        'description': None,
        'intro_content': None,  # Raw markdown body
        'intro_metadata': {},   # Frontmatter metadata
    }

    if not passage_dir.exists():
        return extras

    # Try language-specific intro first, fall back to default
    intro_path = None
    if lang != 'en':
        lang_intro_path = passage_dir / f"intro-{lang}.md"
        if lang_intro_path.exists():
            intro_path = lang_intro_path
    if intro_path is None:
        intro_path = passage_dir / "intro.md"

    if intro_path.exists():
        post = frontmatter.load(intro_path)
        extras['name'] = post.get('name')
        extras['description'] = post.get('description')
        extras['intro_content'] = post.content
        extras['intro_metadata'] = dict(post.metadata)

    return extras


def render_intro(slug: str, passage_dir: Path, content: str, metadata: dict) -> tuple[str, list]:
    """
    Render intro.md content with media helpers.

    Returns (rendered_html, files_to_copy).
    """
    if not content or not content.strip():
        return None, []

    from jinja2 import Template

    renderer = IntroRenderer(slug, passage_dir)

    # Render Jinja2 template with frontmatter vars and media helpers
    body_template = Template(content)
    rendered_body = body_template.render(
        **metadata,
        image=renderer.image,
        audio=renderer.audio,
    )

    # Convert markdown to HTML
    intro_html = render_markdown(rendered_body)

    return intro_html, renderer.files_to_copy


def load_about_content(lang: str = 'en') -> str:
    """Load and render about.md content for specified language."""
    # Try language-specific file first, fall back to default
    if lang != 'en':
        about_path = DATA_DIR / f"about-{lang}.md"
        if about_path.exists():
            with open(about_path, 'r', encoding='utf-8') as f:
                return render_markdown(f.read())
    about_path = DATA_DIR / "about.md"
    if about_path.exists():
        with open(about_path, 'r', encoding='utf-8') as f:
            return render_markdown(f.read())
    return "<p>Edit data/about.md to add content here.</p>"


def create_jinja_env(i18n_strings: dict = None) -> Environment:
    """Create Jinja2 environment with custom filters and i18n."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True
    )

    # Add custom filters
    env.filters['superscripts'] = render_superscripts
    env.filters['slugify'] = slugify

    # Add custom globals
    env.globals['get_valid_sentences'] = get_valid_sentences

    # Add translation helper
    strings = i18n_strings or {}
    def t(key: str, **kwargs) -> str:
        """Translation helper: t('key') or t('key', count=5)"""
        text = strings.get(key, key)
        for k, v in kwargs.items():
            text = text.replace(f'{{{k}}}', str(v))
        return text
    env.globals['t'] = t

    return env


def build_language_version(
    lang: str,
    base_url: str,
    output_dir: Path,
    all_languages: list,
    config: dict,
    passages_data: dict,
    i18n_strings: dict,
):
    """Build a single language version of the site."""
    passages = passages_data.get('passages', [])
    site_title = config.get('site_title', 'Language Passages')
    site_description = config.get('site_description', 'Interactive linguistic text corpus')
    current_version = config.get('version', 'V4')
    track_display_order = config.get('track_display_order', [])

    # Load language-specific content
    about_content = load_about_content(lang)
    glossary_content = load_glossary_content(lang)

    # Build passage metadata with language-specific extras
    passage_list = []
    for i, p in enumerate(passages):
        slug = slugify(p.get('name', f'passage-{i}'))
        extras = load_passage_extras(slug, lang)
        name = extras['name'] or p.get('name', f'Passage {i+1}')
        description = extras['description'] or p.get('description', '')
        passage_list.append({
            'index': i,
            'name': name,
            'description': description,
            'slug': slug,
            'sentence_count': len(get_valid_sentences(p)),
            'intro_content': extras['intro_content'],
            'intro_metadata': extras['intro_metadata'],
        })

    # Create Jinja environment with i18n
    env = create_jinja_env(i18n_strings)

    # Filter to other languages (not current)
    other_languages = [l for l in all_languages if l['code'] != lang]

    # Common template context
    common_context = {
        'site_title': site_title,
        'site_description': site_description,
        'base_url': base_url,
        'lang': lang,
        'other_languages': other_languages,
        'passages': passages,
        'passage_list': passage_list,
        'current_version': current_version,
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    lang_prefix = f"[{lang}] " if lang != 'en' else ""

    # Render about page (index.html)
    template = env.get_template('about.html')
    html = template.render(**common_context, about_content=about_content, current_page='about')
    (output_dir / "index.html").write_text(html, encoding='utf-8')
    print(f"{lang_prefix}Built: index.html")

    # Render passages list
    template = env.get_template('passages.html')
    html = template.render(**common_context, current_page='passages')
    passages_out_dir = output_dir / "passages"
    passages_out_dir.mkdir(parents=True, exist_ok=True)
    (passages_out_dir / "index.html").write_text(html, encoding='utf-8')
    print(f"{lang_prefix}Built: passages/index.html")

    # Render individual passage pages
    template = env.get_template('passage.html')
    for i, passage in enumerate(passages):
        meta = passage_list[i]
        passage_out_dir = passages_out_dir / meta['slug']
        passage_out_dir.mkdir(parents=True, exist_ok=True)
        passage_data_dir = PASSAGES_DATA_DIR / meta['slug']

        # Render intro content with media helpers
        intro_html, files_to_copy = render_intro(
            meta['slug'],
            passage_data_dir,
            meta['intro_content'],
            meta['intro_metadata'],
        )

        # Copy media files referenced in intro
        for src_path, dest_name in files_to_copy:
            dest_path = passage_out_dir / dest_name
            if not dest_path.exists():  # Avoid re-copying for FR version
                shutil.copy2(src_path, dest_path)

        # Build render context
        render_meta = {**meta, 'intro_html': intro_html}
        available_tracks = get_available_tracks(passage, track_display_order)

        html = template.render(
            **common_context,
            passage=passage,
            passage_meta=render_meta,
            available_tracks=available_tracks,
            current_page='passages'
        )
        (passage_out_dir / "index.html").write_text(html, encoding='utf-8')
        print(f"{lang_prefix}Built: passages/{meta['slug']}/index.html")

    # Render search page
    template = env.get_template('search.html')
    html = template.render(
        **common_context,
        passages_json=json.dumps(passages_data, ensure_ascii=False),
        current_page='search'
    )
    search_dir = output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "index.html").write_text(html, encoding='utf-8')
    print(f"{lang_prefix}Built: search/index.html")

    # Render glossary page
    if glossary_content:
        template = env.get_template('glossary.html')
        html = template.render(
            **common_context,
            glossary_content=glossary_content,
            current_page='glossary'
        )
        glossary_dir = output_dir / "glossary"
        glossary_dir.mkdir(parents=True, exist_ok=True)
        (glossary_dir / "index.html").write_text(html, encoding='utf-8')
        print(f"{lang_prefix}Built: glossary/index.html")


def build_site(base_url: str = "/", version_override: str = None):
    """Build the complete static site in all languages."""
    # Load configuration
    config = load_config()
    if version_override:
        config['version'] = version_override

    # Load and filter passages data
    passages_data = load_data()
    passages_data = filter_tracks(passages_data, config)
    passages = passages_data.get('passages', [])

    # Load i18n strings
    i18n = load_i18n()

    # Check for required tools
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found. Install it with:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        raise SystemExit(1)

    if not check_imagemagick():
        print("Error: ImageMagick is required but not found. Install it with:")
        print("  Ubuntu/Debian: sudo apt install imagemagick")
        print("  macOS: brew install imagemagick")
        raise SystemExit(1)

    # Ensure passage data directories exist
    ensure_passage_dirs(passages)

    # Clean and create output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Copy static files
    if STATIC_DIR.exists():
        for item in STATIC_DIR.iterdir():
            dest = OUTPUT_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    # Get language configuration
    languages = config.get('languages', ['en'])
    default_lang = config.get('default_language', languages[0] if languages else 'en')

    # Build language info list with URLs
    all_languages = []
    for lang in languages:
        lang_strings = i18n.get(lang, {})
        if lang == default_lang:
            lang_url = base_url
        else:
            lang_url = f"{base_url}{lang}/"
        all_languages.append({
            'code': lang,
            'url': lang_url,
            'name': lang_strings.get('lang.name', lang.upper()),
            'flag': lang_strings.get('lang.flag', ''),
        })

    # Build each language version
    for lang_info in all_languages:
        lang = lang_info['code']
        if lang == default_lang:
            lang_output_dir = OUTPUT_DIR
        else:
            lang_output_dir = OUTPUT_DIR / lang

        build_language_version(
            lang=lang,
            base_url=lang_info['url'],
            output_dir=lang_output_dir,
            all_languages=all_languages,
            config=config,
            passages_data=passages_data,
            i18n_strings=i18n.get(lang, {}),
        )

    print(f"\nSite built to {OUTPUT_DIR}/")


class RebuildHandler(FileSystemEventHandler):
    """File system event handler with debounced rebuild."""

    def __init__(self, base_url: str, project_dir: Path, version_override: str = None, debounce_ms: int = 20):
        self.base_url = base_url
        self.project_dir = project_dir
        self.version_override = version_override
        self.debounce_ms = debounce_ms
        self.timer = None
        self.lock = threading.Lock()

    def _trigger_rebuild(self):
        """Actually perform the rebuild."""
        print("\nRebuilding...")
        try:
            # Run build from project directory
            original_cwd = os.getcwd()
            os.chdir(self.project_dir)
            build_site(base_url=self.base_url, version_override=self.version_override)
            os.chdir(original_cwd)
        except Exception as e:
            print(f"Rebuild failed: {e}")

    def _schedule_rebuild(self):
        """Schedule a rebuild with debouncing."""
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(
                self.debounce_ms / 1000.0,
                self._trigger_rebuild
            )
            self.timer.start()

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_rebuild()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_rebuild()

    def on_deleted(self, event):
        if not event.is_directory:
            self._schedule_rebuild()


def serve(port: int = 8000, base_url: str = "/", version_override: str = None):
    """Serve the built site locally with file watching."""
    project_dir = Path.cwd()
    serve_dir = (project_dir / OUTPUT_DIR).resolve()

    # Set up file watcher
    handler = RebuildHandler(base_url=base_url, project_dir=project_dir, version_override=version_override)
    observer = Observer()

    watch_paths = [TEMPLATES_DIR, DATA_DIR, STATIC_DIR]
    for path in watch_paths:
        if path.exists():
            observer.schedule(handler, str(path), recursive=True)
            print(f"Watching: {path}/")

    observer.start()

    # Custom handler that serves from absolute path (survives rebuilds)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress request logging

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}/")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping...")
            observer.stop()

    observer.join()


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Build linguistic passages static site")
    parser.add_argument('--serve', '-s', action='store_true', help='Serve site after building')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Port for local server')
    parser.add_argument('--base-url', '-b', default='/', help='Base URL for the site')
    parser.add_argument('--version', '-v', choices=['V1', 'V2', 'V3', 'V4'],
                        help='Override version from config (V1=audio only, V2=translations, V3=+IPA, V4=+gloss)')
    args = parser.parse_args()

    build_site(base_url=args.base_url, version_override=args.version)

    if args.serve:
        serve(port=args.port, base_url=args.base_url, version_override=args.version)


if __name__ == "__main__":
    main()
