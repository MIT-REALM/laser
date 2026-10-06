# Asset and content provenance

Source manuscript directory: `/home/songyuan/Documents/Paper/NeurIPS2026-LASER`.
Figures and content were read from the current local manuscript on 2026-09-30. The manuscript sources were not edited for this website.

| Website asset | Manuscript source |
| --- | --- |
| `media/laser.pdf` | `camera_ready.pdf` |
| `media/motivation.png` | `complex/motiv/motiv_main.pdf` |
| `media/algorithm.png` | `figs/algo.pdf` |
| `media/performance.png` | `figs/main_exp/success_performance_profile.pdf` |
| `media/entropy-tradeoff.png` | `figs/ent-q-tradeoff.pdf` |
| `media/entropy-frontier.png` | `figs/ent_q_line_plot.pdf` |
| `media/ablation-temperature.png` | `figs/ablations/q_tau_curves.pdf` |
| `media/ablation-adjoint.png` | `figs/ablations/bptt_curves.pdf` |
| `media/ablation-prior.png` | `figs/ablations/latent_mixture_curves.pdf` |

PNG assets are rendered with Poppler (`pdftoppm -singlefile -scale-to <pixels> -png`). Main figures use 1800 pixels on the longest side; ablations use 1100; the entropy distribution panel and frontier use 1400 and 700, respectively. No experimental data or curves were reconstructed.

The title, author order, affiliations, venue, abstract, and numerical results follow `camera_ready.tex` and the rendered PDF. Main evaluation: 4 environments × 5 tasks × 2 datasets; three seeds per setting. The 34/40 and 13/40 highlights follow Section 5.2. The 3.6× throughput comparison is the paper's adjoint-matching-versus-BPTT ablation, not a claim about every baseline. The support and BPTT qualifications follow the method and implementation discussion.

## Design and fonts

The HTML, CSS, and JavaScript are written for LASER. The REALM favicon (`static/images/favicon.svg` and `favicon.png`) is reused unchanged from the ReFORM project website; the same assets are also used by DGPPO. The visual structure follows the user's ReFORM, Def-MARL, DGPPO, and GCBF+ sites, which credit the Nerfies project-page template. No prior project's results or videos are reused.

Noto Sans Regular and Bold are subsetted from the local `fonts-noto-core` package into WOFF2 for offline use. See `static/fonts/LICENSE.txt` for the font license. The subset covers the site's text, Latin, Greek, and mathematical symbols where supported by the source font; remaining mathematical glyphs use system fallbacks.

Manuscript PDF SHA-256: `31a5c89d2f33cb46a7db5a0f10040ddda9434d71fd394c38e4724e8c61e1ab1d`.

## LASER rollout videos (added 2026-10-06)

Source: `/home/songyuan/Documents/Code/laser/media`, supplied by the user. The MP4 files are copied unchanged; all are 720 × 480, 30 fps, H.264 with yuv420p pixel format and no audio stream. The GIF duplicates are omitted to keep the page lightweight. Poster JPEGs are frames extracted at 0.5 seconds with FFmpeg.

| Website video | Source filename | Duration |
| --- | --- | --- |
| `media/videos/antmaze-large.mp4` | `antmaze-large.mp4` | 17.8 s |
| `media/videos/cube-single.mp4` | `cube-single.mp4` | 2.534 s |
| `media/videos/cube-double.mp4` | `cube-double.mp4` | 6.7 s |
| `media/videos/scene.mp4` | `scene.mp4` | 5.534 s |

Each video has a same-stem `.jpg` poster in `media/videos/`. These are illustrative policy rollouts; no seed, task number, dataset quality, or playback-speed claims are inferred from the filenames.

## Target latent distribution equation

`media/latent-target.svg` is typeset from `media/latent-target.tex`, matching equation (13) in `camera_ready.tex`: the same Computer Modern math, Times roman text, Helvetica sans-serif latent-space labels, and full critic notation. It omits only the manuscript equation number and trailing prose comma. The SVG contains vector glyph paths, so it stays sharp without external fonts or a JavaScript math renderer. The HTML provides an accessible text alternative.

To regenerate, run `latex -interaction=nonstopmode -halt-on-error latent-target.tex` and `dvisvgm --no-fonts --exact --bbox=min --output=latent-target.svg latent-target.dvi` in a temporary directory containing the `.tex` file, then copy the SVG into `media/`.

## Baseline papers and related work

Baseline links follow the manuscript citations and the authors' project pages:

- ReFORM: [paper](https://openreview.net/forum?id=YvFsyRReeN), linked from the [ReFORM project page](https://mit-realm.github.io/reform/).
- DSRL: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799).
- FQL and IFQL: [Flow Q-Learning](https://arxiv.org/abs/2502.02538), linked from the [FQL project page](https://seohong.me/projects/fql/). IFQL is presented in the same paper.
- QAM and QAM-E: [Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234), linked from the [QAM project page](https://colinqiyangli.github.io/qam/). QAM-E is a variant in the same paper.

The Related Work section follows the supplied screenshot's simple heading-and-prose layout. Its ReFORM summary follows the project page's bounded latent-noise and reflected-flow formulation and refers specifically to learned support.
