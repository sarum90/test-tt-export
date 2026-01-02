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


def get_valid_sentences(passage: dict) -> list:
    """Filter out empty sentences from a passage."""
    sentences = []
    for s in passage.get('sentences', []):
        tracks = s.get('word_tracks', [])
        if tracks and tracks[0].get('words') and len(tracks[0]['words']) > 0:
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


def load_passage_extras(slug: str) -> dict:
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

    # Load intro.md with frontmatter
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


def load_about_content() -> str:
    """Load and render about.md content."""
    about_path = DATA_DIR / "about.md"
    if about_path.exists():
        with open(about_path, 'r', encoding='utf-8') as f:
            return render_markdown(f.read())
    return "<p>Edit data/about.md to add content here.</p>"


def create_jinja_env() -> Environment:
    """Create Jinja2 environment with custom filters."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True
    )

    # Add custom filters
    env.filters['superscripts'] = render_superscripts
    env.filters['slugify'] = slugify

    # Add custom globals
    env.globals['get_valid_sentences'] = get_valid_sentences

    return env


def build_site(base_url: str = "/"):
    """Build the complete static site."""
    passages_data = load_data()
    about_content = load_about_content()
    passages = passages_data.get('passages', [])

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

    # Build passage metadata for templates
    passage_list = []
    for i, p in enumerate(passages):
        slug = slugify(p.get('name', f'passage-{i}'))
        extras = load_passage_extras(slug)
        # Use frontmatter values, fall back to passages.json
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

    # Create Jinja environment
    env = create_jinja_env()

    # Common template context
    common_context = {
        'site_title': 'Language Passages',
        'site_description': 'Interactive linguistic text corpus with interlinear glosses',
        'base_url': base_url,
        'passages': passages,
        'passage_list': passage_list,
    }

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

    # Render about page (index.html)
    template = env.get_template('about.html')
    html = template.render(**common_context, about_content=about_content, current_page='about')
    (OUTPUT_DIR / "index.html").write_text(html, encoding='utf-8')
    print("Built: index.html")

    # Render passages list
    template = env.get_template('passages.html')
    html = template.render(**common_context, current_page='passages')
    passages_out_dir = OUTPUT_DIR / "passages"
    passages_out_dir.mkdir(parents=True, exist_ok=True)
    (passages_out_dir / "index.html").write_text(html, encoding='utf-8')
    print("Built: passages/index.html")

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
            shutil.copy2(src_path, passage_out_dir / dest_name)

        # Build render context
        render_meta = {**meta, 'intro_html': intro_html}

        html = template.render(
            **common_context,
            passage=passage,
            passage_meta=render_meta,
            current_page='passages'
        )
        (passage_out_dir / "index.html").write_text(html, encoding='utf-8')
        print(f"Built: passages/{meta['slug']}/index.html")

    # Render search page
    template = env.get_template('search.html')
    html = template.render(
        **common_context,
        passages_json=json.dumps(passages_data, ensure_ascii=False),
        current_page='search'
    )
    search_dir = OUTPUT_DIR / "search"
    search_dir.mkdir(parents=True)
    (search_dir / "index.html").write_text(html, encoding='utf-8')
    print("Built: search/index.html")

    print(f"\nSite built to {OUTPUT_DIR}/")


class RebuildHandler(FileSystemEventHandler):
    """File system event handler with debounced rebuild."""

    def __init__(self, base_url: str, project_dir: Path, debounce_ms: int = 20):
        self.base_url = base_url
        self.project_dir = project_dir
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
            build_site(base_url=self.base_url)
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


def serve(port: int = 8000, base_url: str = "/"):
    """Serve the built site locally with file watching."""
    project_dir = Path.cwd()
    serve_dir = (project_dir / OUTPUT_DIR).resolve()

    # Set up file watcher
    handler = RebuildHandler(base_url=base_url, project_dir=project_dir)
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
    args = parser.parse_args()

    build_site(base_url=args.base_url)

    if args.serve:
        serve(port=args.port, base_url=args.base_url)


if __name__ == "__main__":
    main()
