# LASER project website

Static project page for **LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL**, on the `website` branch of `MIT-REALM/laser`.

## Preview locally

```bash
cd /home/songyuan/Documents/Website/laser
python3 -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000>. There is no build step or package installation. Fonts, videos, figures, scripts, styles, and the paper are local; external requests happen only when following external links.

## Files

- `index.html`: paper information, overview, method, experiments, ablations, abstract, and citation.
- `static/css/index.css`: responsive layout, following the group's ReFORM, Def-MARL, DGPPO, and GCBF+ project-page style.
- `static/js/index.js`: accessible figure enlargement, citation copying, research-menu dismissal, and motion-aware demo playback. The page and image links also work without JavaScript.
- `media/laser.pdf`: copy of the current manuscript's `camera_ready.pdf`.
- `media/*.png`: figures rendered directly from the manuscript PDFs; see `ASSETS.md`.
- `media/videos/`: four LASER MP4 rollouts and their poster frames, with source details in `ASSETS.md`.

The intended project URL is <https://mit-realm.github.io/laser/>. All local links are relative so the page works both at a server root and under `/laser/`. `.nojekyll` allows direct static serving by GitHub Pages.

## Updating content

Replace `media/laser.pdf` and the corresponding images when the paper changes. Update the text, summary numbers, metadata, and BibTeX in `index.html` together. The current Paper button links to the local PDF; a confirmed arXiv or OpenReview URL can be substituted later. The Code button points to the repository and does not imply that the code has been released.

The teaser gallery contains LASER rollouts for AntMaze-large, Cube-single, Cube-double, and Scene. Videos loop silently with native playback and fullscreen controls. Autoplay is disabled for visitors who prefer reduced motion; without JavaScript, visitors can start playback using the controls. The gallery uses four columns on desktop, two on tablets, and one on phones.

The existing repository remote is preserved. This website was prepared locally without pushing or enabling hosting.

## Validation

Checked in the local browser at desktop, 768 px, 390 px, and 320 px widths. Verified local asset and fragment links, all eight figure loads, figure-dialog open/close, keyboard Escape dismissal, research navigation, and BibTeX clipboard copying. No page-level horizontal overflow or browser warnings/errors were observed. The bundled manuscript PDF is byte-identical to the source.

Video-gallery update (2026-10-06): verified playback of all four MP4s, keyboard pause/resume, the Videos anchor, and layouts at 1280, 768, 390, and 320 px. No horizontal overflow or browser warnings/errors were observed. Copied MP4 hashes match their source files.
