# go-concepts: why loops look alive to Go AIs

Code, generated positions and results for the paper

> **Loops Look Alive: The Shared Misjudgement Behind the Cyclic Attack on Go AIs**
> Jaeson Booker, 2026. The LaTeX source is in [`paper/`](paper/).

Self-play Go engines that play far above human level lose to the "cyclic" attack of
Wang et al. (2023): the adversary builds a closed loop of the victim's stones around its
own group and captures it. This repository asks what inside the networks the attack
exploits. In short: the vulnerable networks read doomed loops as alive, because a
"closed loop" feature built by mid-network keeps evidence of death from building on the
loop. Holding that one direction fixed flips the verdict, in raw readouts and inside a
tree search. A KataGo network trained against the attack keeps the feature and reads
doomed loops correctly anyway.

Everything runs on CPU. The full record of every experiment, with the exact numbers
quoted in the paper, is [`docs/build-log.md`](docs/build-log.md).

## Layout

| Path | Contents |
|---|---|
| `goconcepts/` | the package: exact labels, position generators, network adapters, probes, interventions, search |
| `tests/` | label tests against KataGo's own board code |
| `data/` | generated position sets (pickles of `goconcepts` objects), all built from fixed seeds |
| `runs/` | result files (JSON, logs, small readouts); large activation dumps are not included |
| `paper/` | LaTeX source of the paper and `make_figures.py` |
| `docs/build-log.md` | experiment-by-experiment log, run by run |
| `make_*.py`, `run_*.sh` | scripts that build the position sets and run each network |

## Setup

Python 3.10+ with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

**KataGo's Python code.** The adapters use the board, features and model code from
KataGo's `python/` directory. Clone KataGo (we used commit `3c144b3e797185caa98e2328ae69757f36d791ba`)
into `ext/KataGo`, or point `KATAGO_PY` at its `python/` directory:

```
git clone https://github.com/lightvector/KataGo ext/KataGo
```

**Networks.** Put these files in `nets/` (or set `GOCONCEPTS_NETS` to their directory).
They are not redistributed here.

| File name expected | Network | Source |
|---|---|---|
| `g170e-b20.bin.gz` | g170e-b20c256x2 (2020 g170 run) | KataGo GitHub releases |
| `b40-s11840935168.bin.gz` | kata1-b40c256-s11840935168-d2898845681, the net FAR attacked | katagotraining.org |
| `kata1-tf3-b11c768-s11001M-d5973M.ckpt` | kata1-tf3-b11c768-s11001M-d5973M | katagotraining.org |
| `humanv0.ckpt` | b18c384nbt-humanv0 | KataGo GitHub releases |
| `lz_best.gz` | Leela Zero final 40x256 network | leela-zero project |

g170 and b40 are read from KataGo's `.bin.gz` format by our own loader
(`goconcepts/kgbin_adapter.py`); the other KataGo nets are PyTorch checkpoints.

**External game records** (only for the real-game results, Section 5 of the paper):

- FAR's published attack games, from <https://goattack.far.ai>, compiled into
  `data/far/far_adversarial_games.txt.gz` (one game per line: key, header JSON, setup
  stones, colours, moves). See `goconcepts/realgames.py:load_far`.
- Fox professional games: `Pro.7z` and `Pro2.7z` from
  <https://github.com/featurecat/go-dataset>, unpacked into `data/sgf/` (or set `GOCONCEPTS_SGF`).
- The public cyclic test positions from <https://github.com/AlignmentResearch/go_attack>
  go in `data/cyclic_sgfs/`.

**Speed note.** On some CPUs, tf3's PyTorch forward pass runs about 40 times slower than it
should because of denormal floating-point values. `goconcepts/katago_adapter.py` calls
`torch.set_flush_denormal(True)`, which gives identical outputs at 0.4 s a position.

## Reproducing the paper

Run from the repository root. The position sets in `data/` are already built; the
`make_*.py` scripts rebuild them from their seeds.

| Paper section | Command |
|---|---|
| Tests | `python -m pytest -q tests` |
| 3. Races | `python make_races.py && ./run_races.sh && python -m goconcepts.race_eval score runs/humansl runs/lz runs/g170 runs/tf3` |
| 4. Group safety | `python make_status_balanced.py 40 && ./run_status_bal.sh` (then `status_eval score ... --cases data/status_bal.pkl`) |
| 4. Hopeless loops | `python make_status_hopeless.py && python -m goconcepts.hopeless_eval --engine kgbin --ckpt nets/g170e-b20.bin.gz --out runs/g170/hopeless.json` |
| 5. Real games | `python -m goconcepts.realgames build && ./run_real.sh && ./run_b40.sh` |
| 6. Loop direction holds | `python -m goconcepts.mech_cross --net b40 --blocks 12-24 --fit-on safe` |
| 6. Verdict path, death channels, gating | `python -m goconcepts.verdict_path`, `python -m goconcepts.late_channels2 --net b40 --blocks 38-40 --mid 12-24`, `python -m goconcepts.gating --net b40 --mid 12-24` |
| 7. Closure wave and channels | `python -m goconcepts.loopcircuit wave --net g170`, `python -m goconcepts.closure_channels --net g170 --upto 12`, `python -m goconcepts.closure_hold --net b40 --blocks 12-24` |
| 7. Closure means alive | `python -m goconcepts.tf3_closure dirs`, `... cos`, `... hold --blocks 2-8`, `python -m goconcepts.closure_swap --net b40 --blocks 7-29` |
| 8. New shapes | `python -m goconcepts.shapegen && python -m goconcepts.shape_eval --net tf3` |
| 8. Automated search, fortified walls | `python -m goconcepts.fortified_set build`, `python -m goconcepts.attack_search run --gens 0`, `python -m goconcepts.attack_search flip --n-seeds 20 --steps 10` |
| 9. Search | `python -m goconcepts.mcts --set --net b40 --visits 800`, add `--hold loop` or `--hold random` |
| Figures | `PYTHONPATH=. python paper/make_figures.py` |

`docs/build-log.md` gives the exact command, settings and output file for every result.

## Notes

- Ownership readouts are from the chain owner's side after the owner passes (+1 lives,
  -1 dies), read straight off each network's ownership head.
- Every direction and probe is fit on one half of the positions (fixed seed) and tested on
  the other.
- The race labels assume the opponent walls cannot be profitably attacked. In the original
  generated sets the walls were weak; `data/status_bal_fortified.pkl` strengthens them
  (Section 8 of the paper), and the main results hold on it.
- The tree search in `goconcepts/mcts.py` is our own simplified PUCT search, not KataGo's
  engine.

## Citation

```
@article{booker2026loops,
  title  = {Loops Look Alive: The Shared Misjudgement Behind the Cyclic Attack on Go AIs},
  author = {Booker, Jaeson},
  year   = {2026}
}
```
