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

The HTML, CSS, JavaScript, and favicon are written for LASER. The visual structure follows the user's ReFORM, Def-MARL, DGPPO, and GCBF+ sites, which credit the Nerfies project-page template. No prior project's results or videos are reused.

Noto Sans Regular and Bold are subsetted from the local `fonts-noto-core` package into WOFF2 for offline use. See `static/fonts/LICENSE.txt` for the font license. The subset covers the site's text, Latin, Greek, and mathematical symbols where supported by the source font; remaining mathematical glyphs use system fallbacks.

Manuscript PDF SHA-256: `31a5c89d2f33cb46a7db5a0f10040ddda9434d71fd394c38e4724e8c61e1ab1d`.
