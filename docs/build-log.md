# goconcepts

Code for *Testing Concept Convergence in Self-Play Go Agents*. This is the start of the
cheapest useful version: stage one (layerwise probes) plus stage three (the cyclic probe),
with KataGo as the first engine.

## Layout

| Path | What it does |
|---|---|
| `goconcepts/labels.py` | Exact labels from the board: chains, liberties, Benson pass-alive, eye space, ladders. Uses no engine. |
| `goconcepts/sgf.py` | Minimal SGF main-line reader (Fox pro games). |
| `goconcepts/cyclic.py` | Loads lightvector's cyclic-group suite (105 cases, 36 with CORRECT answer moves, 63 with a target winner). |
| `goconcepts/katago_adapter.py` | Replays games into KataGo's GameState, encodes inputs, hooks all 18 trunk blocks plus the stem. |
| `goconcepts/extract.py` | Builds `natural.npz` (sampled stones from pro games, split by game) and `cyclic.npz` (every stone in every cyclic case). |
| `goconcepts/probes.py` | Linear probes per layer per concept. Scored against exact labels only. Resumes after an interruption. |
| `goconcepts/lz_adapter.py` | Leela Zero text weights into a PyTorch tower (stones-only inputs), same interface as the KataGo adapter. |
| `goconcepts/cyclegen.py` | Procedural cyclic/cut minimal pairs: a ring enclosing an opponent group vs the same ring cut by one stone. |
| `goconcepts/pairs_eval.py` | Extraction and paired evaluation on the minimal pairs, four probe variants, bootstrap CIs. |
| `goconcepts/kgbin_adapter.py` | KataGo `.bin.gz` models (versions 8-10, e.g. the 2020 g170 nets on GitHub) parsed into PyTorch, following KataGo's C++ desc and Eigen backend. |
| `goconcepts/pairs_shared.py` | First shared-failure test (best layer per engine; superseded, kept for the record). |
| `goconcepts/pairs_layeravg.py` | Shared-failure test with gaps averaged over all trunk blocks, within-liberty permutation tests, partial correlations controlling for HumanSL. |
| `goconcepts/racegen.py` | Solved capturing races: a ring (or an edge-pocket control chain) against an enclosed group, walled in so an exact minimax settles it; kept only when some move captures and passing loses the ring. |
| `goconcepts/race_eval.py` | Races through each engine: policy mass on the capturing moves, probe-D liberty readout of the same chain, behaviour-representation links, shared failures. |
| `goconcepts/statusgen.py` | Doomed vs safe D chains at matched liberty counts (2-5), ring and edge geometry, labelled by exact search after a D pass. `generate_balanced`: the eye-balanced set (one real inner eye in every case, D liberties 5 or 7 in both classes). |
| `goconcepts/status_eval.py` | Safety readouts: ownership and value after a hypothetical pass, edge-trained safety probe applied to rings, link to capture behaviour. |
| `goconcepts/realgames.py` | Ownership after a pass on chains that were really captured: FAR's published cyclic-attack games vs ordinary captures in the same games and in Fox pro games, by lead time and liberties. |
| `run_b40.sh` | The attacked net (`kata1-b40c256-s11840935168`, FAR's "Latest") through real games, both status sets and the races. |
| `goconcepts/mech.py` | Mechanism of the loop flip in g170: block patching, gradient x update attribution by board region, and one-direction holds per block (fit on doomed or on safe positions). |
| `goconcepts/hopeless_eval.py` | Ownership of ring and enclosed group on hopeless loops (two-eyed inner group, ring with only outside liberties). |
| `goconcepts/mech_cross.py` | The loop-direction hold across g170, b40 and tf3 (hooks for KataGo's PyTorch blocks), fit on safe or doomed positions. |
| `goconcepts/verdict_path.py` | Doomed rings' position on each layer's verdict axis, per-block write decomposition along the final axis, and the causal patch that removes the loop's verdict writes at chosen blocks. |
| `goconcepts/late_channels.py`, `late_channels2.py` | Per-channel split of the last blocks' verdict writes (b40, g170); zeroing and restoring the "death" channels; whether holding the loop feature re-lights them. |
| `goconcepts/gating.py` | Death-evidence trajectory per layer, per-block loop holds inside the mid window, and where the released death evidence is written (b40, g170). |
| `goconcepts/loopcircuit.py`, `closure_hold.py` | Minimal-pair closure wave by layer and distance, region blocking, and the pure-closure hold on the eye-balanced set. |
| `goconcepts/steer.py` | Causal test on g170: add the edge-trained "doomed" direction to the D chain's activations and measure capture. |
| `goconcepts/compare.py` | Cross-engine report: depth curves, size-matched cyclic test, stage-two error correlation. |
| `tests/test_labels.py` | Liberties, ladders and pass-alive checked against KataGo's own Board code on real pro positions. |
| `data/far/far_adversarial_games.txt.gz` | 116 games from goattack.far.ai (`/sgfs/<set>/game_infos.csv` and the SGFs it lists), moves only: one line per game with key, header, setup stones, colours, moves. |
| `data/cyclic_sgfs/` | Copied from AlignmentResearch/go_attack `scripts/runcycletestsandplot/sgfs`. |

## Running

```
python -m pytest -q tests
python -m goconcepts.extract --games 300 --per-game 4 --points 24 --out runs/humansl
python -m goconcepts.probes --run runs/humansl
ENGINE=lz CKPT=nets/best-network OUT=runs/lz ./run_all.sh
python -m goconcepts.compare runs/humansl runs/lz
./run_pairs.sh   # minimal-pair cyclic test, both engines
./run_g170.sh    # KataGo g170 self-play net, all stages
./run_tf3.sh     # post-2023 KataGo, reduced run (slow on CPU)
./run_matched.sh # refit probe D for the other engines on tf3's training data
python -m goconcepts.pairs_layeravg --tag m60 runs/lz runs/g170 runs/humansl runs/tf3
python make_races.py && ./run_races.sh
python -m goconcepts.race_eval score runs/humansl runs/lz runs/g170 runs/tf3
python make_status.py 25 && ./run_status.sh
python -m goconcepts.status_eval score runs/g170 runs/humansl runs/lz runs/tf3
python -m goconcepts.steer --layers 3,6,8 --alphas 0,8,16,32
python -m goconcepts.realgames build && ./run_real.sh
python -m goconcepts.realgames score runs/g170 runs/humansl runs/tf3
python make_status_balanced.py 40 && ./run_status_bal.sh
python -m goconcepts.status_eval score runs/g170 runs/humansl runs/lz runs/tf3 --cases data/status_bal.pkl --npz status_bal.npz --out runs/statusbal_report.json
```

External inputs: KataGo's `python/` directory (set `KATAGO_PY`), a `.ckpt` checkpoint,
and a folder of 19x19 SGFs. The Fox pro set is `Pro.7z` and `Pro2.7z` from
featurecat/go-dataset (10,348 games).

## Concepts and what KataGo is already given

KataGo's input planes are not just stones. That changes how its probe results can be read.

| Concept | Label | KataGo input leak | How the code handles it |
|---|---|---|---|
| Liberties | exact, bucketed 1..8+ | planes 3-5 mark chains with 1, 2 or 3 liberties | report accuracy on 4+ separately; include the `input` layer as a baseline |
| Group membership | exact, pairwise same-chain | none | elementwise-product pair probe |
| Pass-alive (Benson) | exact | planes 18-19 under area scoring with no tax | run under Japanese rules, where those planes are blank |
| Eye space | exact: enclosed empty regions vital to the chain, up to 7 points | none | 0 / 1 / 2+ |
| Ladder | exact search, budgeted | planes 14-17 | report, but treat as supervised-by-input for KataGo |
| Territory | not built yet | planes 18-19 plus the ownership head | needs finished games |

Leela Zero and ELF get stones and history only. A cross-engine comparison on liberties
or ladders is therefore input-confounded for KataGo, the same way the proposal already
flags the ownership target.

## Access notes (cloud sandbox)

Reachable: GitHub (KataGo releases, including `b18c384nbt-humanv0.ckpt` and older
`g170` `.bin.gz` nets; ELF OpenGo v2 weights), PyPI.
Blocked: katagotraining.org, zero.sjeng.org (Leela Zero weights), goattack.far.ai,
huggingface.co. The go_attack GitHub repo holds the cyclic suite. goattack.far.ai works from the desktop browser (see the eighth run).

## First run: KataGo HumanSL (b18c384nbt-humanv0), CPU

1,200 positions from 300 Fox pro games (28,800 sampled stones, split 80/20 by game) and
all 105 cyclic cases (12,537 stones, 71 cyclic chains). Full numbers in
`runs/humansl/results.json`. Probe accuracy against exact labels:

| Concept | input | block3 | block9 | block18 |
|---|---|---|---|---|
| Liberties, natural held-out | 0.56 | 0.78 | 0.74 | 0.70 |
| Liberties 4+, natural held-out | 0.44 | 0.72 | 0.67 | 0.63 |
| Liberties, stones in cyclic chains | 0.29 | 0.64 | 0.48 | 0.36 |
| Same chain (pairwise, balanced), natural | 0.52 | 0.86 | 0.87 | 0.85 |
| Same chain, pairs touching a cyclic chain | 0.59 | 0.90 | 0.94 | 0.94 |
| Pass-alive (balanced), natural | 0.79 | 0.92 | 0.97 | 0.92 |
| Ladder (balanced), natural | 0.99 | 1.00 | 0.99 | 0.99 |

HumanSL puts 0.54 of its raw policy mass on the CORRECT moves in the 36 answer-move cases.

How to read this:

1. Ladders are solved at the input layer because KataGo is given ladder planes. The
   concept tells us nothing about KataGo. It becomes informative only for Leela Zero and ELF.
2. Liberty readout peaks at blocks 2 to 5 and declines with depth. On cyclic chains it
   declines much faster.
3. **That gap is mostly a chain-size effect, not a cyclic effect.** At block 15, cyclic
   chains with 2 liberties read correctly 0.37 of the time (per-chain mean, 8 chains).
   Non-cyclic chains of 10+ stones with 2 liberties in the same positions read 0.38. At 3
   liberties the cyclic chains are worse (0.48 against 0.68, 12 chains). Natural pro positions
   contain few large chains, so the probe never trained on them. The next run has to
   size-match (`chain_size` is now stored) and report per chain, not per stone.
4. Stones in one chain are not independent samples. Effective n for cyclic results is the
   71 chains, and fewer once you split by liberty bucket.
5. In HumanSL, pairwise group membership does not degrade on cyclic chains. HumanSL is the
   human reference, not an attacked self-play engine, so this says nothing yet about the
   three hypotheses. It does show the pair probe can see cyclic connectivity when it is there.

## Second run: Leela Zero (40 blocks x 256, final SWA net)

Same 1,200 positions, same 28,800 sampled stones, same cyclic cases. Loader check: the raw
policy matches the pro's move 68% of the time on 60 positions. Full report in
`runs/compare_humansl_lz.json`.

| | HumanSL | Leela Zero |
|---|---|---|
| Liberties, input layer | 0.56 (planes given) | 0.34 |
| Liberties, best layer | 0.78 at block 2 of 18 | 0.64 at block 3 of 40 |
| Liberties 4+, best layer | 0.72 | 0.58 |
| Same chain, best layer | 0.87 | 0.78 |
| Ladder, best layer | 1.00 (planes given) | 0.96 at block 10, falls to 0.82 by block 40 |
| Policy mass on CORRECT cyclic answers | 0.54 | 0.16 |

What this run shows:

1. **The behavioural gap is real and large.** On lightvector's answer-move cases Leela Zero's
   raw policy puts 0.16 on the correct moves, HumanSL 0.54. The engine FAR says is vulnerable
   fails these positions, the human-trained net mostly doesn't.
2. **Leela Zero learns liberties, but shallow and early.** Readout peaks at block 3 and decays
   through the remaining 37 blocks (0.42 at block 40). Both engines compute liberties early
   and then stop carrying them in linearly readable form.
3. **The size-matched cyclic test is suggestive for Leela Zero, not conclusive.** At block 3,
   2-liberty cyclic chains read correctly 0.06 of the time against 0.41 for large non-cyclic
   chains (8 vs 9 chains). 4 liberties: 0.10 vs 0.34. But 3 liberties shows no gap (0.21 vs
   0.18). HumanSL shows no cyclic-specific gap at its best layer. With 6 to 12 chains per cell,
   none of this survives a significance test yet.
4. **Stage two: most shared error is shared difficulty.** Raw, both engines are wrong on the
   same held-out stone 2.1 times as often as independence predicts. Stratified by liberty
   bucket and chain size, the ratio falls to 1.17. The engines agree mostly because the same
   stones are hard, not because they make the same mistake. That residual 1.17 is the number
   to track across engines.

Where this leaves the three hypotheses: coverage gap and shared wrong concept are both still
live. The run weighs against "knowledge not used" for Leela Zero at the linear level, because
the liberty information on cyclic chains isn't there to be unused.

## Third run: cyclic vs cut minimal pairs

The lightvector suite has 71 cyclic chains, and reflections or colour swaps of it don't
add independent samples: both engines were trained on all 8 symmetries and see the board
relative to the side to move. So `cyclegen.py` builds new ones. Each pair is two positions
that differ by one stone. In the **cyclic** member, a ring chain encloses an opponent group
and has a target liberty count from 2 to 8. In the **cut** member, one ring stone is replaced
by an opponent stone, so the ring is an open C of the same size, shape and surroundings that
encloses nothing. 400 pairs, 358 with identical liberties in both members; evaluation uses
the 179 matched pairs in the held-out half. Eight context stones are replayed as the last
moves so both engines see a full move history (see the history note below).

The primary probe (D) is trained on natural positions plus both variants from the training
half, so it has seen closed and open rings equally. Metric: fraction of a chain's stones
whose liberty bucket is read correctly, as a paired difference cut minus cyclic.

| Layer | Leela Zero cut | LZ cyclic | LZ gap (95% CI) | HumanSL cut | HumanSL cyclic | HumanSL gap (95% CI) |
|---|---|---|---|---|---|---|
| early (LZ b4, HS b4) | 0.34 | 0.28 | +0.07 [0.03, 0.10] | 0.57 | 0.53 | +0.05 [0.01, 0.08] |
| mid (LZ b10, HS b10) | 0.39 | 0.27 | +0.12 [0.07, 0.15] | 0.59 | 0.57 | +0.02 [-0.01, 0.06] |
| late (LZ b30, HS b15) | 0.30 | 0.18 | +0.12 [0.08, 0.16] | 0.57 | 0.57 | +0.01 [-0.03, 0.04] |
| last (LZ b40, HS b18) | 0.28 | 0.18 | +0.10 [0.06, 0.14] | 0.57 | 0.56 | +0.01 [-0.03, 0.04] |

By liberty count, Leela Zero at block 8:

| Liberties | pairs | cut | cyclic |
|---|---|---|---|
| 2 | 34 | 0.82 | 0.51 |
| 3 | 33 | 0.42 | 0.14 |
| 4 | 32 | 0.30 | 0.10 |
| 5 | 21 | 0.44 | 0.49 |
| 6 | 26 | 0.19 | 0.11 |
| 7 | 16 | 0.19 | 0.22 |
| 8+ | 17 | 0.29 | 0.53 |

What this shows:

1. **In Leela Zero, closing the loop degrades the liberty representation.** Same chain,
   same liberties, one stone different, and the fair probe reads the closed version 10 to 14
   points worse at every layer from block 6 to block 40. The gap is concentrated at 2 to 4
   liberties, the regime where the cyclic attack actually captures. At 5 and 8+ liberties the
   cyclic chains read as well or better, so this is not a blanket penalty on loops.
2. **It survives the training-distribution check.** Probe C, trained only on cyclic chains
   from the training half, still reads cut chains better (+0.07 [0.03, 0.12] at block 8).
   Probe B, trained on cut chains, widens the gap to +0.23. So the gap is not an artefact of
   which chains the probe saw.
3. **HumanSL shows essentially no gap once the probe is fair.** Probe D gives +0.00 to
   +0.05 across layers, with the CI crossing zero from block 6 on. At 2 and 3 liberties this
   is guaranteed, because HumanSL gets 1/2/3-liberty input planes. At 4 liberties HumanSL
   does show a gap (0.46 vs 0.24, 32 pairs), so it is not immune where the planes stop.
4. **The earlier HumanSL "gap" was a probe artefact.** Natural-only probe A shows +0.07 at
   blocks 12 to 15. It vanishes under probe D, so it came from the probe never seeing large
   rings, not from the network.
5. **History matters for Leela Zero.** The first version of these pairs was setup-only, so
   seven of Leela Zero's eight history planes were zero. Its liberty readout then sat near
   chance (0.13 to 0.19) on both variants. With eight replayed moves, the cut chains read at
   0.39. Any Leela Zero result on setup positions needs this check. The lightvector cases have
   5 to 13 history boards, so they are only partly affected.

For the three hypotheses: in Leela Zero the information is degraded, so "knowledge not used"
is out for this engine, at the linear level. The result fits both "shared wrong concept" and
"coverage gap". Those two differ only on whether a second self-play engine fails on the
**same** pairs. `pairs.npz` keeps per-pair results, so that test runs as soon as a KataGo
self-play net is in.

Caveats: linear per-stone probes only. 22 layers per engine were tested, but the Leela Zero
gap is consistent across all of them from block 4 on rather than one lucky layer. Per-bucket
cells hold 16 to 34 pairs.

## Fourth run: KataGo self-play (g170e-b20c256x2, 2020) and the shared-failure test

katagotraining.org is blocked here and the transfer from your computer kept timing out, so
this uses the g170 self-play nets that KataGo's GitHub releases still carry. They predate
KataGo's anti-cyclic training (2023), so they are the plain AlphaZero-style KataGo the attack
was built against. `kgbin_adapter.py` loads the C++ `.bin.gz` format directly. Checks: the
parser consumes the file to the last byte, and the raw policy matches the pro's move 62% of
the time on 60 positions.

g170 has the same input planes as HumanSL (liberties 1-3, ladders), a plain 20-block
residual tower, and self-play training. So **g170 vs HumanSL holds the inputs fixed and varies
the training data**, which is the cleanest contrast in the project so far.

| | HumanSL | g170 (KataGo self-play) | Leela Zero |
|---|---|---|---|
| Policy mass on CORRECT cyclic answers | 0.54 | **0.04** | 0.16 |
| Liberties, best layer (natural) | 0.78 | 0.69 | 0.64 |
| Minimal-pair gap, probe D, layer chosen on training pairs | +0.01 [-0.02, 0.05] | +0.04 [0.00, 0.07] | +0.12 [0.08, 0.16] |
| Gap at 2 / 3 / 4 liberties | +0.01 / +0.08 / +0.16 | +0.00 / +0.02 / +0.28 | +0.32 / +0.25 / +0.18 |

Shared-failure test, first version (`runs/pairs_shared.json`, **superseded below**): it chose one
layer per engine by probe-D accuracy on the training pairs. Probe D was fit on those pairs, so
the choice was in-sample and favoured overfit layers. That version reported LZ ~ g170 at
r = 0.19, falling to 0.01 after controlling for HumanSL, and I read it as "no self-play-specific
sharing". The fifth run corrects this and reverses that reading.

What still holds from this run:

1. **The attacked self-play KataGo fails the cyclic positions badly** (0.04 on the correct
   moves), worse than Leela Zero and far worse than HumanSL.
2. **Nets read closed loops worse at the liberty counts they have to compute themselves.**
   Leela Zero gets no liberty planes and fails at 2 to 4. The two KataGo nets are handed 1-3
   liberties as input and barely fail there; both fail at 4.

## Fifth run: post-2023 KataGo (kata1-tf3-b11c768) and the corrected shared-failure test

`kata1-tf3-b11c768` is an 11-block hybrid: each block mixes convolution with RoPE attention
(768 channels). It was trained after KataGo added cyclic positions to its training, so it is
the adversarially trained counterpart of g170. KataGo's own Python code loads it. The trunk
runs in sequence layout, which the adapter converts back to board layout; the trunk
cumulative check passes (max error 0.014).

It costs about 15 seconds per position on this CPU, so it got a reduced run: 200 natural
positions, all held-out test pairs and 60 training pairs. For a like-for-like comparison,
probe D was refit for every other engine on exactly the same reduced training data
(`--restrict-train-to runs/tf3`). And the per-engine layer choice is gone: each pair's gap is
averaged over every trunk block (`pairs_layeravg.py`), so no step looks at test pairs or
rewards overfitting. 179 held-out matched pairs.

| | HumanSL | g170 (2020 self-play) | tf3 (post-2023, adversarially trained) | Leela Zero |
|---|---|---|---|---|
| Policy mass on CORRECT cyclic answers | 0.54 | 0.04 | **0.99** | 0.16 |
| Cyclic gap, all blocks (95% CI) | +0.03 [0.00, 0.06] | +0.05 [0.03, 0.07] | **+0.05 [0.02, 0.08]** | +0.08 [0.06, 0.11] |
| Gap, middle third of the trunk | +0.03 | +0.05 | +0.06 | +0.10 |
| Middle third at 2 / 3 / 4 liberties | .02 / .15 / .12 | .03 / .15 / .25 | .00 / .06 / .28 | .22 / .20 / .20 |

Which chains fail (layer-averaged per-pair gaps, Spearman, p from permutations within
liberty-count strata):

| Engines | r | p | controlling for HumanSL's gap: r | p |
|---|---|---|---|---|
| Leela Zero ~ g170 | 0.39 | <0.001 | **0.30** | <0.001 |
| g170 ~ tf3 | 0.40 | <0.001 | **0.22** | 0.001 |
| Leela Zero ~ tf3 | 0.29 | <0.001 | **0.17** | 0.011 |
| each ~ HumanSL | 0.31 to 0.49 | <0.001 | | |

With each engine's full, unmatched training data (`runs/pairs_layeravg_none.json`) the picture
is similar. The Leela Zero gap is larger (+0.12), Leela Zero ~ g170 controlling for HumanSL is
unchanged (0.31), and g170 ~ tf3 controlling for HumanSL drops to 0.04 (p = 0.29). So the
tf3 part of the sharing result depends on the matched refit and is the least secure number here.

What this shows:

1. **Adversarial training fixed the behaviour, not the representation.** tf3 plays the cyclic
   positions almost perfectly (0.99, up from g170's 0.04), but its liberty readout on closed
   loops is as degraded as g170's (+0.05 vs +0.05), concentrated at 4 liberties (+0.28 mid-trunk).
   The patch sits on top of an unchanged representation, or routes around it.
2. **The self-play engines share a failure the human net doesn't.** After removing everything
   that tracks HumanSL's gap, the three self-play nets still agree on which cyclic chains they
   misread (partial r 0.17 to 0.30). That's across a 40-block ResNet, a 20-block ResNet with
   global pooling, and a conv-attention hybrid: two independent training projects, two input
   encodings, three architectures. The Leela Zero ~ g170 link is the most robust (0.30 under
   both training setups).
3. **HumanSL has the smallest gap** (+0.03, CI touching zero), with the same input planes as
   g170 and tf3. With inputs held fixed, human versus self-play training data is the difference
   that tracks the gap (architectures differ too, so this isn't fully isolated).
4. **Correction.** The previous run's "no self-play-specific sharing" came from choosing
   layers in-sample. With that removed, the sharing is clearly there.

Where this leaves the three hypotheses:

- **Knowledge not used**: out. The liberty information on loops is degraded in all three
  self-play nets, even the one that now plays these positions correctly.
- **Shared wrong concept**: now the best-supported. Self-play nets share which loops they
  misread, beyond the human net, across three architectures.
- **Coverage gap**: weakened. Adding cyclic positions to training fixed play but left the
  representation gap unchanged. If missing data were the whole story, the representation
  should have improved too.

For the proposal's safety point, this is the uncomfortable case: an audit of tf3's play would
find the blind spot closed, while its internal liberty representation on loops is as wrong as
the vulnerable net's.

Caveats:

- Linear per-stone probes of liberty count only; a non-linear or group-level readout could see
  more.
- tf3 was probed with less training data. The matched refit puts every engine on the same
  footing, but all gaps are measured with that smaller probe.
- **tf3's 0.99 may be contaminated.** lightvector wrote both the test suite and KataGo's
  training, and KataGo's anti-cyclic training may include positions like these. A fresh
  behavioural test on generated positions is needed before calling the behaviour fixed in
  general.
- Partial correlations control for HumanSL's gap linearly, in ranks. Shared difficulty that
  HumanSL doesn't register could still contribute.

## Sixth run: play on fresh, solved capturing races

`racegen.py` builds capturing races with an exact answer. A D chain races an enclosed A
group. The D chain's outside liberties are walled in by A stones, so extending gains nothing
and the race is closed; an exact minimax (capture > seki > loss) settles it. A position is
kept only if some D move captures the inner group and playing elsewhere loses the D chain.
D is to move and A's last move is next to the chain, as in a game. Six liberty structures
(D chain with 1 to 5 liberties, inner group with 0 to 2 eyes), 24 positions each, in two
geometries:

- **cyclic**: the D chain is a closed ring around the inner group
- **edge**: the inner group sits in a pocket of a D chain and the board edge, so nothing is
  enclosed by one chain; same counting, different shape

288 positions, generated from a seed, so no engine has seen them. The generator solves every
position; the test suite re-solves a sample after replaying it through KataGo's board code, to
catch any mismatch between the stored answer and the position the engines actually see. Score: raw-policy
mass on the capturing moves. Representation: probe D (trained on the same reduced data for
every engine) reads the liberty count of every stone of the same D chain, averaged over
blocks. "Cyclic effect" is cyclic minus edge after regressing out liberty structure and chain
size (edge chains are smaller: median 14 vs 22 stones).

| | HumanSL | Leela Zero | g170 (2020 self-play) | tf3 (post-2023) |
|---|---|---|---|---|
| Mass on capture, cyclic | 0.37 | 0.02 | 0.10 | 0.62 |
| Mass on capture, edge | 0.52 | 0.31 | 0.74 | 0.70 |
| Cyclic effect on play (95% CI) | -0.25 [-0.32, -0.18] | -0.33 [-0.42, -0.24] | **-0.66 [-0.75, -0.56]** | -0.11 [-0.25, 0.04] |
| Cyclic effect on liberty readout | -0.11 [-0.15, -0.06] | -0.10 [-0.15, -0.05] | **+0.01 [-0.03, 0.05]** | **-0.14 [-0.19, -0.11]** |
| Within-engine link, readout ~ play | -0.01 | 0.01 | -0.07 | -0.06 |

g170 by D-chain liberties, cyclic vs edge: 1: 0.58 vs 0.95, 2: 0.00 vs 0.74, 3: 0.00 vs 0.65,
4: 0.00 vs 0.69, 5: 0.00 vs 0.76. On cyclic races its policy mass goes to neither race points
nor the area of the last move (0.02) but elsewhere on the board. It plays as if the race
weren't there.

What this shows:

1. **The blind spot is real on fresh positions, and huge in the 2020 KataGo.** g170 finds the
   capture in 74% of edge races and essentially never in a cyclic race with 2 or more
   liberties. Leela Zero fails cyclic races almost completely too (0.02). No contamination
   is possible here.
2. **Adversarial training generalises.** tf3 captures in cyclic races at 0.62, close to its
   edge rate (0.70). Its 0.99 on lightvector's suite was high but not memorisation: the fix
   carries over to positions it can't have seen, with a remaining dip at 5 liberties (0.40
   vs 0.72).
3. **Play and liberty readout come apart.** g170 has the worst cyclic play and *no* cyclic
   deficit in its liberty readout on the same chains. tf3 has the largest readout deficit and
   the best cyclic play. At 1 liberty g170 is even told the chain is in atari by an input
   plane, and still misses the capture 42% of the time on rings. Within each engine, positions
   where the readout is worse are not the positions where play is worse (all |r| < 0.07).
4. **Engines fail the same positions.** Controlling for geometry, liberty structure and size,
   per-position play correlates across engines: g170 ~ tf3 0.59, humansl ~ tf3 0.42,
   Leela Zero ~ g170 0.41 (all p < 0.01).

Where this leaves the three hypotheses, revised:

- **Knowledge not used**: back in, and now the best description of the behavioural failure in
  KataGo nets. g170 carries the liberty count on these rings as well as on open chains (and is
  handed it as input at 1 to 3) but doesn't act on it when the chain is a loop.
- **Shared wrong concept**: holds for the liberty representation (fifth run), but that shared
  representational flaw doesn't explain play. The concept that goes wrong is downstream of
  liberty counting, most likely a learned judgement of group status (a ring enclosing enemy
  stones looks alive), which these probes don't measure.
- **Coverage gap**: the tf3 result supports it for behaviour. Adding cyclic positions to
  training fixed play on fresh cyclic races while leaving the liberty readout gap in place.
  The fix went in at the same downstream stage where the failure lives.

For the safety argument: the failure and its fix both live where a liberty probe doesn't
look. Auditing the "right" concept (liberties) would have said g170 was fine on rings and tf3
was worse. The next probe should target status and urgency directly.

Caveats: raw policy without search. The edge control matches the counting, not the shape or
size, and edge-of-board play differs from centre play. The within-engine link uses a linear
per-stone liberty probe only; a status probe could link where this one doesn't. Races are
walled in, which is cleaner than real games.

## Seventh run: does the net know the group is doomed?

The sixth run showed play and liberty readout come apart. This run tests the next concept up:
group safety. Label (exact search): **doomed** if the D chain is captured when D plays
elsewhere, **safe** if D still wins the race after passing. 400 positions: ring vs edge geometry
x doomed vs safe x D-chain liberties 2 to 5, 25 per cell. Within a cell the liberty count is
the same, so liberties can't give the status away. (What does differ is the inner group:
doomed cells mostly have an inner eye, safe cells mostly don't.)

**Readout 1: the net's own ownership prediction after a hypothetical pass.** D passes, the
net predicts end-of-game ownership of the D chain from D's side (+1 survives, -1 captured).
This is the label's own counterfactual, read straight off the ownership head, no probe.

| Ownership after a pass | g170 (2020) | HumanSL | tf3 (post-2023) |
|---|---|---|---|
| Edge: doomed / safe | **-0.63 / +0.73** | -0.44 / +0.55 | -0.67 / +0.76 |
| Ring: doomed / safe | **+0.76 / +0.73** | +0.45 / +0.63 | **-0.75 / +0.74** |
| Ring: separates doomed from safe (AUC) | 0.66 | 0.80 | **0.95** |
| Ring: safe minus doomed (95% CI) | -0.04 [-0.19, 0.11] | +0.18 [0.05, 0.29] | +1.48 [1.31, 1.63] |

g170 on doomed rings by liberties: 2: +0.11, 3: +0.99, 4: +0.97, 5: +0.99. It is certain that
rings with 3 or more liberties survive when they don't, while it reads the same fate correctly
for edge groups. Leela Zero has no ownership head, and the value head after a pass is
uninformative for every engine here (AUC about 0.5), because the random rest of the board
dominates the win probability.

**Readout 2: a safety probe trained on edge groups, applied to rings.** Share of doomed rings
it calls safe (safe rings called doomed is 0.00 to 0.11 for every engine at every block):

| Block depth | g170 | HumanSL | tf3 | Leela Zero |
|---|---|---|---|---|
| early | 0.00 to 0.28 | 0.07 to 0.74 | 0.35 to 0.74 | 0.23 |
| middle | 0.62 to 0.85 | 0.73 to 0.91 | **0.06 to 0.28** | 1.00 |
| late | **1.00** | 0.60 to 0.70 | **0.04 to 0.17** | 1.00 |

In g170 and Leela Zero, rings drift to the "safe" side as depth increases and end there. In
tf3 they start there and are pulled back from block 6 on. A probe trained on rings themselves
decodes ring status well in every engine (AUC 0.97 to 1.00; Leela Zero falls to 0.81 by its
last block), but that probe can
use the inner group's eye, which differs between cells, so it doesn't show the net knows.

**Readout 3: does safety drive play?** Within doomed rings, correlation between ownership
after a pass and policy mass on the capture (liberty levels removed): tf3 **r = -0.57,
p < 0.001** (the more doomed it reads the ring, the more it captures). g170 -0.05 (its capture
rate is 0.003, no room to vary), HumanSL +0.10 (n.s.).

**Readout 4: causal steering in g170.** Adding the edge-trained "doomed" direction to the ring's
stones, scaled in standard deviations of edge activations along it:

| Where, how much | Doomed rings: capture | Safe rings | Doomed edge groups | Random direction, doomed rings |
|---|---|---|---|---|
| none | 0.003 | 0.013 | 0.62 | 0.003 |
| block 3, x16 | 0.16 | 0.02 | 0.65 | 0.014 |
| block 6, x16 | **0.32** | 0.15 | 0.35 | 0.012 |
| block 8, x32 | 0.48 | 0.50 | 0.19 | 0.09 |

At moderate strength (block 3, x16) the push is specific: capture on doomed rings rises about
50-fold, safe rings barely move, edge groups are untouched, and a random push does nothing.
Stronger pushes help doomed rings more but spill over into safe rings (capturing there is
still sound) and disrupt edge play.

What this shows:

1. **g170 has the wrong safety concept for loops.** Its own ownership head says doomed rings
   live (about +0.98 at 3 to 5 liberties) while it judges edge groups correctly. Internally,
   rings drift to "safe" with depth. Pushing that one direction partly restores the capture.
2. **tf3's fix is in the safety concept.** Its ownership head reads rings as correctly as
   edges, its internal bias on rings is removed from block 6 on, and its safety reading
   predicts its play position by position. The liberty readout gap from the fifth run is still
   there; the adversarial training corrected the judgement built on top of it.
3. **HumanSL sits in between** on every readout: partial safety reading on rings, partial play.
4. **Leela Zero's internal safety probe looks like g170's**, rings read safe at every depth past
   block 8, but it has no ownership head to confirm it.

This refines the sixth run's reading. g170 does carry the liberty count on rings, but the
knowledge its decision depends on, "this group dies", is absent, not ignored. Where the
three hypotheses stand:

- **Shared wrong concept**: best supported, located one level above liberties. The self-play
  nets misjudge loop safety; the human net partly does; the adversarially trained net doesn't.
- **Knowledge not used**: out for g170 at the level that matters. Its safety judgement is wrong,
  and its play follows that judgement.
- **Coverage gap**: consistent with tf3. Training on cyclic positions fixed the safety concept
  and, with it, the play.

For the safety argument in the proposal: auditing the concept the decision actually uses (group
safety) catches g170 and clears tf3, correctly. Auditing the obvious lower-level concept
(liberties) gets both wrong. In this case interpretability works, provided the audit targets
the right level of abstraction.

Caveats: doomed and safe cells differ in the inner group's eyes, so the internal probe results
could partly reflect eye detection; the ownership results don't depend on a probe. Steering
strengths are large (16 to 32 standard deviations). Raw policy, no search. Positions are walled,
artificial races with a random board around them.

## Eighth run: real attack games, and the eye confound removed

Two checks on the seventh run: does "loops look alive" show up where the attack actually
happens, and does it survive once the inner group's eyes can't give the answer away?

### Real games

FAR's site serves its game files under `goattack.far.ai/sgfs/<set>/` (blocked from the
sandbox, so they came through the desktop browser). 116 games: the cyclic adversary against
KataGo from 1 visit to 10M visits (cp505, cp505h, b18, b60), humans playing the attack
against KataGo and Leela Zero, and transfer games against Leela Zero and ELF. Every chain of
6 or more stones that gets captured is an event. For each event, positions 1, 3, 5, 9, 17
and 33 moves before the capture, with the losing side to move. The readout is the seventh
run's: that side passes, and the net predicts ownership of the chain (+1 lives, -1 dies).
Every chain here did in fact die.

- **Loops:** 73 victim chains that enclose an opponent group of 4+ stones at that point
  (median 42 stones), one per game.
- **Ordinary chains:** 376 captured chains that don't (median 8 stones): the rest of the FAR
  games plus 2,000 Fox pro games. No loop was captured in any pro game.

Ownership after a pass, chains that really died, 3 or more moves before the capture:

| Chain liberties | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| g170: loops | +0.57 | **+0.99** | **+1.00** | **+1.00** | **+1.00** |
| g170: ordinary | -0.51 | -0.56 | -0.61 | -0.34 | -0.16 |
| HumanSL: loops | -0.59 | -0.41 | -0.19 | +0.28 | +0.38 |
| HumanSL: ordinary | -0.48 | -0.51 | -0.57 | -0.35 | -0.15 |
| tf3: loops | -0.92 | -0.85 | -0.63 | +0.14 | |
| tf3: ordinary | -0.61 | -0.42 | -0.84 | +0.01 | |

(n for loops: 144, 42, 16, 10, 10 positions; tf3 ran on a subset, leads 3, 9 and 17.)

By lead time, loops vs ordinary chains:

| Moves before capture | 1 | 3 | 9 | 17 | 33 |
|---|---|---|---|---|---|
| g170: loops read alive | 3% | 58% | **75%** | **78%** | **83%** |
| g170: ordinary read alive | 2-6% | 10-13% | 14-18% | 18-21% | 22-31% |
| HumanSL: loops read alive | 0% | 12% | 17% | 35% | 46% |
| tf3: loops read alive | | 1% | 10% | 27% | |

Adjusted for liberties, log size and lead time, being a loop shifts the reading by
**+0.69 [0.57, 0.80]** for g170, -0.04 [-0.14, 0.08] for HumanSL and -0.29 [-0.66, 0.09]
for tf3. Size doesn't explain g170's result: ordinary chains of 20+ stones still read dead
(-0.40 to -0.19 at 3 to 9 moves out).

g170 sees the loop die only when the capture is one move away (-0.69). Before that it is
sure the loop lives. That is the window the attack uses. The chain was killable for many
moves, and the victim spent them elsewhere. g170 isn't one of the attacked nets, but it's the
same training family, and the synthetic results predicted this. HumanSL and tf3 read real
loops as dead as ordinary chains, or deader.

### The eye confound, removed

In the seventh run's set, doomed cells mostly had an inner eye and safe cells mostly didn't.
The first fix matched the nominal race parameters (shared liberties, eyes, outside
liberties) and failed. Within a matched cell, status flipped on whether the "eye" touched the
shared liberty, so whether it was a real eye. That set is kept as
`data/status_bal_nominal.pkl`, for the record.

The final set (`data/status_bal.pkl`, 320 positions) gives every inner group exactly one real
eye, a point surrounded by inner stones only. The D chain has the same liberty count in both
classes, 5 or 7. What decides the status is how D's liberties split between shared and
outside ones. Against an eye, shared liberties count for the eyed side. So D, to move, wins
only if outside >= shared + 1, and survives a pass only if outside >= shared + 2. Doomed is
(2,1,3) or (3,1,4). Safe is (1,1,4), (2,1,5) or (1,1,6), written as (shared, eyes, outside).
40 per cell: ring and edge, doomed and safe, 5 and 7 liberties. Checked: real eyes 1.0 in
every cell. Ring libs are equal, and the wall's weakest chain and ring and inner sizes are
matched. The inner group's liberty count still differs (shared + 1), and that's inherent to
the race.

| Ownership after a pass (doomed / safe) | g170 | HumanSL | tf3 |
|---|---|---|---|
| Edge | -0.33 / +0.97 | -0.32 / +0.76 | -0.55 / +0.98 |
| Ring | **+0.99 / +0.99** | +0.55 / +0.85 | **-0.68 / +0.98** |
| Ring AUC | 0.83 | 0.98 | **1.00** |
| Capture mass, doomed ring / doomed edge | **0.00** / 0.55 | 0.17 / 0.31 | **0.52 / 0.54** |
| Ownership predicts capture, doomed rings | r = -0.13 (n.s.) | -0.10 (n.s.) | **-0.74, p < 0.001** |

Leela Zero: capture 0.01 on doomed rings, 0.16 on doomed edge groups.

Edge-trained safety probe applied to rings, share of doomed rings called safe (safe rings
called doomed: 0.00 to 0.11 everywhere past block 2):

| Engine | Blocks 1-2 | Where rings flip to "safe" | Last blocks |
|---|---|---|---|
| g170 | 0.00 | block 4 (0.66), block 5 on (1.00) | **1.00** |
| Leela Zero | 0.00 | block 4 (0.93), block 6 on (1.00) | **1.00** |
| HumanSL | 0.00 to 0.04 | block 3 (0.95) to block 11 | 0.06 to 0.70, falling |
| tf3 | 0.61 to 0.84 | | **0.05 to 0.12 from block 7** |

With eyes held fixed, the seventh run's picture holds and gets sharper. In the first two
blocks, g170, Leela Zero and HumanSL all place doomed rings with doomed edge groups. The
evidence is there early. By block 5 (g170) or 6 (Leela Zero), every doomed ring has moved to
"safe" and stays there. HumanSL moves too but partly comes back. tf3 starts muddled and pulls
doomed rings back to "doomed" from block 7 on. In tf3, ownership predicts capture, position
by position.

A within-ring probe still decodes status almost perfectly in every engine (AUC 0.96 to
1.00; Leela Zero down to 0.76 at its last block). With eyes fixed, that probe probably reads
the inner group's liberties (2 vs 3, visible in KataGo's input planes). So it shows the race
is decodable, not that the net judges it.

What this adds:

1. **"Loops look alive" is what the attack exploits.** In real attack games g170 reads doomed
   loops as certain to live until one move before capture. It reads equally doomed ordinary
   chains, including big ones, as dying. HumanSL and tf3 don't make this error.
2. **The eye confound didn't produce the seventh run's result.** Matched on real eyes and D's
   liberties, g170 still reads every doomed ring as alive and never captures it. The internal
   flip to "safe" happens between blocks 3 and 5, after early blocks got it right.
3. **tf3's fix holds on both sets.** Its safety reading is correct and drives its play.

Caveats: g170, HumanSL and tf3 weren't the victims in these games (cp505, b18, b60, Leela Zero
and ELF were). Real loops are bigger than ordinary captured chains, and no pro game had a loop
captured, so the loop comparison leans on the FAR games. tf3's real-game readout covers a
subset: all loops at three lead times, 123 ordinary-chain positions. Raw nets, no search.

## Ninth run: the net that was actually attacked

`kata1-b40c256-s11840935168` is the network FAR's paper calls Latest (cp505 in their game
files). It was KataGo's strongest net when the attack was built, and the "hardened" games use
the same weights with a search-side fix. It came from katagotraining.org through the desktop
browser (sha256 `4179c907...62bc`). It is a model version 10 `.bin.gz`, so `kgbin_adapter.py`
loads it unchanged. The parser consumes every byte, and the raw policy matches the pro's move
on 47% of 60 positions (g170: 52% on the same positions). 43 of the 73 real loop captures are
this net's own losses (sets `no_search_hardened`, `4096_visits_hardened`, `10mil_visits`,
`win-rate`, `position-analysis`, `position-analysis-1b`, `qualitative`).

**Its own lost loops, ownership after a pass** (chains that really died):

| Moves before capture | 1 | 3 | 5 | 9 | 17 | 33 |
|---|---|---|---|---|---|---|
| b40: reads the loop alive | **40%** | **77%** | 50% | **88%** | **92%** | **91%** |
| b40: mean | -0.25 | +0.49 | +0.11 | +0.70 | +0.75 | +0.80 |
| g170 on the same loops | 2% | 65% | 31% | 68% | 67% | 76% |
| Ordinary captured chains, b40 | 1-7% | 9-13% | 9-11% | 14-18% | 18-21% | 22-33% |

By liberties (all real loops, 3+ moves out): b40 reads loops with 2 liberties at +0.83 and
3 to 8 liberties at +1.00, against -0.19 to -0.61 for ordinary chains at 2 to 6 liberties.
Adjusted for liberties, size and lead time, being a loop shifts b40's reading by
**+0.88 [0.76, 1.00]** (g170: +0.69).

One move before the capture the loop is in atari, and KataGo is told so by an input plane.
b40 still reads 40% of those loops as alive after a pass (mean -0.28 at 1 liberty). Three
moves out, with 2 liberties, it reads them at +0.97. That fits FAR's own win-rate trace for
this net, which swings back above 99% a few moves before the capture.

**Synthetic sets.** Same readouts as the sixth and eighth runs:

| | g170 | **b40 (attacked)** | tf3 |
|---|---|---|---|
| Races: capture mass, ring / edge | 0.10 / 0.74 | **0.19 / 0.71** | 0.62 / 0.70 |
| Races, ring at 3 / 4 / 5 liberties | 0.00 / 0.00 / 0.00 | 0.06 / 0.07 / 0.03 | 0.67 / 0.58 / 0.40 |
| Status set: doomed ring / doomed edge, ownership after a pass | +0.76 / -0.63 | **+0.69 / -0.64** | -0.75 / -0.67 |
| Eye-balanced: doomed ring / doomed edge | +0.99 / -0.33 | **+0.66 / -0.51** | -0.68 / -0.55 |
| Eye-balanced: safe ring | +0.99 | +0.92 | +0.98 |
| Eye-balanced: capture, doomed ring / doomed edge | 0.00 / 0.55 | **0.03 / 0.51** | 0.52 / 0.54 |
| Eye-balanced: edge-trained probe calls doomed rings safe | 1.00 from block 5 | **1.00 from block 6**, 0.94 to 1.00 to block 40 | 0.05 to 0.12 from block 7 |

b40 behaves like g170, slightly less extreme on synthetic rings. Its ownership does rank
doomed rings below safe ones (AUC 0.95 on the eye-balanced set, against 0.83 for g170), but
the whole ring distribution sits on the "alive" side. Within doomed rings, the more doomed
it reads a ring the more it captures (r = -0.33, p < 0.001), so the reading and the play
are linked, and both are wrong.

What this settles:

1. **The attacked net has the blind spot the stand-in showed.** On its own lost games, b40
   reads doomed loops as living, more confidently than g170 (+0.88 vs +0.69 adjusted), while
   it reads doomed ordinary chains as dying. The attack works because this judgement is
   wrong.
2. **The same internal flip.** Its edge-trained safety probe files doomed rings with doomed
   edge groups at block 2, then as safe from block 6 to block 40.
3. **The chain from concept to play holds in the victim:** wrong safety reading on loops,
   no capture on loops (0.03 to 0.07 at 3+ liberties), and the two linked position by
   position.

Caveats: raw net, no search (FAR's games used 1 to 10M visits; search didn't rescue it there
either). The version-10 loader was validated by full parse and policy agreement, not against
KataGo's C++ output on identical inputs.

## Tenth run: what flips a loop to "safe" in g170

Question: the edge-trained safety probe moves doomed rings to "safe" between blocks 3 and 5.
Is that where the error is made, and what carries it? `goconcepts/mech.py`, g170, the
eye-balanced set unless noted. Readouts as before: ownership of the D chain after a D pass
(+1 lives), and policy mass on the capture.

**1. Blocks 3 to 5 don't cause it.** Replacing the D chain's residual updates with the mean
update of doomed edge groups, at any single block from 1 to 12 or any window of up to six
blocks inside 1 to 12, leaves doomed rings at +0.96 to +0.98 and capture at 0.00. Only patching blocks 1
to 12 together moves them (+0.99 to +0.23, capture 0.00 to 0.20). The probe flip at blocks
3 to 5 is where the verdict becomes linearly visible for rings of this size, not a switch.

**2. One direction per block carries it.** For each block, take the unit vector from the
mean trunk state on D stones of doomed edge groups to that of doomed rings (fit on half the
positions, tested on the other half). Hold the D stones' component along it at the edge
level, and nothing else:

| Blocks held | Doomed ring: ownership / capture | Safe ring | Doomed edge | Safe edge |
|---|---|---|---|---|
| none | +0.99 / 0.00 | +0.99 | -0.26 / 0.49 | +0.97 |
| 1-5, 3-5, 1-8 | +0.94 to +0.97 / 0.00 | +0.97 to +0.99 | -0.12 to -0.19 / 0.49 | +0.96 |
| 7-10 | +0.91 / 0.03 | +0.95 | -0.29 / 0.51 | +0.94 |
| **9-12** | **+0.25** / 0.07 | +0.53 | -0.15 / 0.50 | +0.82 |
| **6-12** | **+0.06 / 0.23** | +0.55 | -0.14 / 0.49 | +0.81 |
| 1-12 | +0.09 / 0.21 | +0.50 | -0.12 / 0.47 | +0.80 |
| 13-20 | -0.46 / 0.08 | -0.48 | -0.41 / 0.50 | -0.37 |
| 1-12, random directions (3 draws) | +0.98 / 0.00 | +0.99 | unchanged | unchanged |

Holding one of 256 dimensions per block at blocks 6 to 12 takes doomed rings from certain
life to a coin flip, and the capture rate from 0 to 0.23. Random directions do nothing.
Safe rings stay more alive than doomed ones (+0.55 vs +0.06), so with the direction held,
part of the race reading comes back: the doomed-safe gap goes from 0.00 to 0.49. Blocks 13
to 20 are not specific: there the direction merges with the ownership readout itself, and
every group reads dead.

**3. The direction encodes the loop, not the verdict.** Fit on safe positions only, where
rings and edge groups get the same answer, the direction is almost the same (cosine 0.89 to
0.97 with the doomed-fit one at blocks 6 to 12). Holding that version at blocks 1 to 12 still
takes doomed rings to +0.35 (capture 0.10). Adjacent blocks' directions have cosine near 1,
drifting slowly with depth: one persistent "this chain is a closed loop" feature in the
residual stream. Blocks 9 to 12 are where it takes over the verdict (block 12 is a
global-pooling block; so is block 7).

**4. The loop has to be seen before it can mislead.** On a size-varied set (312 positions,
chains of 10 to 60 stones), the block at which the edge-trained probe flips a doomed ring to
"safe" tracks the ring's size: Spearman 0.87 with stone count, 0.84 with span. Span 6 to 9:
blocks 5 to 7. Span 10 to 13: blocks 13 to 17. Ownership reads every size alive (+0.96 to
+1.00), so large rings still end up judged safe, just later. The median real attack loop
(42 stones, span 10 to 15) sits in the late group. This fits a signal that has to spread
around the chain before the net treats it as a loop.

**5. What the loop feature does to the reading.** It makes the net lose the race for the
enclosed group:

- Doomed rings (the one-eyed inner group actually wins): g170 reads the inner group as dead,
  +0.89 from D's side. For doomed edge groups it reads the inner group alive (-0.52).
- **Hopeless loops** (`data/status_hopeless.pkl`, 180 positions): the inner group has two
  real eyes and shares no liberty with the ring, and the ring has 3 to 5 outside liberties.
  Nothing is close: the ring dies. g170 now reads the inner group correctly as alive
  (-0.94), yet still reads the ring as alive in 58% of cases (mean +0.15). Edge groups:
  -0.92. b40, the attacked net, gets these mostly right (-0.68 vs -0.92 for edges).

So the feature isn't a rule that "enclosed means dead". It is a prior toward the loop living,
strong enough to override a close race, and in g170 strong enough to leave a hopeless loop
half-alive while also seeing the group inside it live.

What this adds:

1. **Blocks 3 to 5 are where the error becomes visible for small rings, not where it's
   made.** The causal window is blocks 6 to 12, mostly 9 to 12.
2. **A single "closed loop" direction, present in doomed and safe rings alike, drives the
   wrong verdict.** Holding it at the edge value restores capture on doomed rings and
   partly restores the race reading, with random directions as control.
3. **The loop signal takes depth to form, and more depth for bigger loops.** The real
   attack builds big loops, which the net resolves late.

Caveats: the intervention holds a mean-difference direction at one value on the chain's
stones only; it doesn't identify the circuit that builds the feature. Held directions also
move safe rings (to +0.55) and edge groups a little, so they aren't a clean single-concept
edit. Ownership after a pass and raw policy only.

## Eleventh run: the loop direction in the attacked net and the fixed net

Same test as the tenth run, made uniform across architectures (`goconcepts/mech_cross.py`).
Per block, the shape direction is fit on **safe** positions only, where rings and edge groups
get the same verdict: from edge groups to rings, on the D chain's stones, separately for
the to-move and after-pass states (KataGo's inputs are relative to the side to move). It is fit
on one half of the eye-balanced set and tested on the other. The hold sets the D stones'
component along it to the doomed-edge level, in the first ~60% of the trunk. Controls:
random directions, same blocks.

**Does each net have the feature?** Yes, all of them, and equally strongly.

- Along the shape direction, held-out rings sit 1.7 to 2.0 pooled standard deviations from
  edge groups in g170, b40 and tf3 alike.
- On the cyclic-vs-cut minimal pairs (same chain, one stone different), a linear probe on
  the ring's stones tells closed from cut loops at 0.95 to 0.99 in every net and at nearly
  every block. That includes g170 (0.99 at block 4), b40 (0.99), tf3 (0.99), HumanSL (0.99)
  and Leela Zero (0.99, falling to 0.89 by block 38).

**Does the verdict depend on it?**

| Net, blocks held | Doomed ring: ownership / capture | Safe ring | Doomed edge | Safe edge |
|---|---|---|---|---|
| g170, none | +0.99 / 0.00 | +0.99 | -0.26 / 0.49 | +0.97 |
| g170, 1-12 | **+0.35 / 0.10** | +0.64 | 0.00 / 0.43 | +0.89 |
| g170, 6-12 | **+0.26 / 0.14** | +0.65 | -0.10 / 0.46 | +0.90 |
| g170, 1-12, random | +0.98 / 0.00 | +0.99 | -0.29 / 0.50 | +0.97 |
| **b40**, none | +0.65 / 0.04 | +0.93 | -0.51 / 0.47 | +0.97 |
| **b40**, 1-24 | **-0.21 / 0.22** | +0.75 | -0.31 / 0.42 | +0.95 |
| **b40**, 12-24 | **-0.23 / 0.26** | +0.61 | -0.46 / 0.44 | +0.96 |
| **b40**, 1-24, fit on doomed | -0.34 / 0.24 | +0.66 | -0.39 / 0.43 | +0.92 |
| **b40**, 1-24, random | +0.63 / 0.03 | +0.93 | -0.50 / 0.47 | +0.97 |
| **tf3**, none | -0.73 / 0.52 | +0.98 | -0.51 / 0.51 | +0.98 |
| **tf3**, 1-7 | **-0.72 / 0.64** | +0.93 | -0.52 / 0.51 | +0.97 |
| **tf3**, 4-7 | -0.73 / 0.49 | +0.98 | -0.51 / 0.51 | +0.98 |
| **tf3**, 1-7, random | -0.74 / 0.52 | +0.98 | -0.51 / 0.52 | +0.98 |

(g170 and b40: 160 held-out positions. tf3: 80, 20 per cell, since it costs about 15 s a
forward pass here.)

1. **In the attacked net, the shape feature sets the wrong verdict, more cleanly than in
   g170.** Holding it at the edge level flips b40's doomed rings from alive (+0.65) to dead
   (-0.21) and raises capture from 0.04 to 0.22. Safe rings stay alive (+0.75). The
   doomed-safe gap on rings grows from 0.28 to 0.96, two thirds of the edge groups' 1.47.
   With the loop signal held down, most of b40's race reading comes back.
   Blocks 12 to 24 carry it, the same relative depth as g170's 6 to 12.
2. **tf3 has the same feature and no longer lets it decide.** Holding it changes tf3's
   verdict on doomed rings by 0.01 (still dead), and safe rings barely move (+0.98 to +0.93).
   Capture on doomed rings even rises a little (0.52 to 0.64). The adversarial training
   didn't remove the loop representation. It removed the loop's pull on the safety judgement.

This is the mechanistic version of the fifth run's puzzle, that tf3 fixed play but kept a
degraded liberty representation on loops. The representation of "this is a closed loop" is
intact in the fixed net, as strong as in the broken ones. What changed is downstream: in
g170 and b40 the loop signal writes "alive" into the safety verdict, in tf3 it doesn't.

For the safety argument: an audit that asks "does the model represent the dangerous feature?"
would find the loop feature in all three nets and can't tell the broken nets from the fixed
one. An audit that asks "does this feature drive the decision?" (hold it and watch the
verdict) separates them correctly: large effect in g170 and b40, none in tf3.

Caveats: the direction is a mean difference between geometries and may carry more than "loop"
(ring stones sit mid-board, edge stones near the edge). The safe-fit version rules out the
verdict itself, not every geometric confound. The minimal-pair probe tells closed from cut
loops but the cut stone is a local change. tf3 was tested on 80 positions.

## Twelfth run: where "loop" becomes "alive"

`goconcepts/verdict_path.py`, eye-balanced set, after-pass states (the ones the ownership
verdict is read from). For each layer, the **verdict axis** is the direction from doomed to
safe edge groups on the D chain's stones. It is fit on edge positions only, so it carries no
ring geometry, and on one half of the positions, then measured on the other.

**Observation 1: in mid-trunk, every net puts rings on the "safe" side.** Doomed rings'
position on each layer's verdict axis (0 = doomed edge group, 1 = safe edge group), every
other layer:

| Net | Early | Middle | Late | Final layer |
|---|---|---|---|---|
| g170 | 0.01, -0.14 | 0.66, **1.67**, 1.34 | 0.99, 0.96, 1.01, 0.90 | 0.98 |
| b40 | 0.15, -0.79, -0.23 | 0.45 to **1.39** (blocks 6-22) | 0.67, **-0.12** (block 26), 0.01 to 0.47 | 0.63 |
| tf3 | 0.15, -0.20 | **0.64** (block 4), 0.29 | 0.21, 0.08 | -0.06 |

All three have a mid-trunk stretch where a doomed loop looks safe, tf3 included (0.86 at
block 3). b40 even pulls rings most of the way back to "doomed" at blocks 24 to 26, then
drifts up again. So the early "loop looks alive" signal is not what separates the nets.

**Observation 2: the vulnerable nets write "alive" onto the readout in their last blocks.**
Decompose the final verdict position of doomed rings into each block's update projected on
the final verdict axis, ring minus edge. The terms sum exactly to the final position (0 =
reads like a doomed edge group).

| Net | All blocks | Last block | Last 2 blocks | Last quarter of the trunk | Everything before |
|---|---|---|---|---|---|
| g170 | +0.99 | **+0.53** | +0.69 | +0.87 | +0.12 |
| b40 | +0.63 | **+0.36** | +0.44 | +0.58 | +0.05 |
| tf3 | -0.06 | **-0.11** | -0.15 | -0.15 | +0.09 |

For safe rings the same late writes are near zero in every net (g170 0.00, b40 -0.10,
tf3 -0.14). The late "alive" write is conditional: g170 and b40 add it exactly when the race
says the loop dies. tf3 adds nothing, or a little toward "doomed".

**Causal test.** At chosen blocks, on the D chain's stones, replace the component of each
block's update along that layer's verdict axis with the mean update edge groups of the same
status get along it. This removes what the block writes onto the verdict because the chain
is a loop, and keeps what it writes because of the race.

| Net, blocks | Doomed ring ownership (baseline) | After removing the loop's verdict writes | Safe ring |
|---|---|---|---|
| g170, 1-6 | +0.99 | +0.98 | +0.99 |
| g170, 7-12 | +0.99 | +0.96 | +1.00 |
| g170, 13-16 | +0.99 | +0.80 | +1.00 |
| g170, 19-20 | +0.99 | **+0.39** | +0.99 |
| g170, 13-20 | +0.99 | **-0.21** | +1.00 |
| b40, 1-9 | +0.65 | +0.65 | +0.93 |
| b40, 10-23 | +0.65 | +0.45 | +0.94 |
| b40, 24-27 | +0.65 | +0.78 (more alive) | +0.99 |
| b40, 38-40 | +0.65 | **-0.11** | +0.95 |
| b40, 28-40 | +0.65 | **-0.48** (doomed edge: -0.49) | +0.98 |
| tf3, 4-11 | -0.73 | -0.65 | +1.00 |

Removing the loop's verdict writes in the last quarter of the trunk makes the vulnerable
nets read doomed loops exactly as they read doomed edge groups. Removing them anywhere
earlier barely matters, even where the mid-trunk bias is largest. b40's blocks 24 to 27 are
a partial correction toward "doomed" that the last blocks overwrite. tf3's late blocks carry
no ring-specific push to remove. Capture moves little under these patches (b40: 0.04 to 0.09),
because the policy is read from the to-move state and its own head.

Putting runs ten to twelve together, the path in the vulnerable nets:

1. **Loop feature** (all nets): built by mid-trunk, as strong in tf3 as in the others.
2. **Conversion** (g170 and b40 only): the last blocks read the loop feature and write
   "alive" onto the verdict axis, only when the race says the loop dies. Holding the loop
   feature at the edge level in the middle (tenth and eleventh runs) removes the input to
   this step. Removing the step's verdict writes (this run) does the same from the other end.
3. **tf3's fix sits at step 2.** Its last blocks don't make the conversion, and its final
   blocks lean slightly the other way (-0.15). The feature is still there and even
   produces the same mid-trunk "looks alive" bump (0.86 at block 3), which never reaches the
   readout.

Caveats: the verdict axis is a mean difference on edge groups, one direction per layer; the
patches use the status label to choose the edge reference, so they test where the
ring-specific component matters, not whether the net could recover status on its own.
The last blocks sit just before the heads, so the late decomposition partly measures the
heads' input format. tf3 was tested on 80 positions.

## Thirteenth run: the conversion is a silenced death detector

`goconcepts/late_channels.py`, `goconcepts/late_channels2.py`. The last blocks of b40 (38 to 40)
and g170 (19, 20), eye-balanced set, after-pass state, fit on one half and tested on the other.

Each block's write is `conv3x3(t, fin)`, where `t` is the block's 256 hidden channels after
ReLU. `fin` is linear, so the write onto the final verdict axis splits exactly over the
hidden channels. I ranked channels by how much of the doomed-loop-minus-doomed-edge verdict
difference each one carries.

**1. It's spread over a few dozen channels, not one.** The top 8 channels carry 48% of b40's
ring-specific late write (g170: 38%), the top 32 carry 87% (g170: 72%).

**2. Most of them are "this group is dying" channels that stay silent on loops.** Of the top
32 writers, 27 in b40 and 29 in g170 are more active on doomed edge groups than on doomed
loops. They don't write "alive" for loops. They write "dead" for groups that are dying,
and they don't fire when the dying group is a loop:

| Channel | Doomed edge | Doomed loop | Safe edge | Safe loop |
|---|---|---|---|---|
| b40 block 40 ch 102 | 10.3 | 0.3 | 0.0 | 0.0 |
| b40 block 39 ch 3 | 9.8 | 0.9 | 0.1 | 0.0 |
| b40 block 40 ch 184 | 10.5 | 1.0 | 0.0 | 0.3 |
| b40 block 40 ch 64 | 7.4 | 0.1 | 0.0 | 0.1 |
| g170 block 20 ch 140 | 7.2 | 0.1 | 0.0 | 0.1 |
| g170 block 20 ch 80 | 7.7 | 0.0 | 0.0 | 0.0 |
| g170 block 19 ch 246 | 4.9 | 0.0 | 0.0 | 0.0 |

(mean activation on the D chain's stones). Across the top 16 death channels, doomed loops get
20% of the doomed-edge activation in b40 and 10% in g170, about what safe groups get. A
handful of the remaining writers are the reverse, "loop" channels that fire on rings of
either status (b40 block 40 ch 84: 12.4 on doomed loops, 8.6 on safe loops, 1.3 on doomed
edges).

**3. The death channels are causal, both ways.** Zeroing the top 32 on the D chain and its
neighbours makes doomed *edge* groups read alive (b40 -0.51 to +0.89, g170 -0.26 to +0.50;
32 random channels: no change), and does nothing to loops, which were silent already.
Restoring them on the loop, set to the activation same-status edge groups get:

| Restored | b40 doomed loop | b40 safe loop | g170 doomed loop | g170 safe loop |
|---|---|---|---|---|
| none | +0.65 | +0.93 | +0.99 | +0.99 |
| 8 death channels | +0.39 | +0.93 | +0.88 | +0.99 |
| 16 death channels | **+0.27** | +0.87 | +0.81 | +0.99 |
| 32 death channels | +0.28 | +0.87 | +0.77 | +0.99 |
| 32 random channels | +0.63 | +0.93 | +0.99 | +0.99 |

In b40, re-lighting 16 channels in the last three blocks closes a third of the distance
between doomed loops and doomed edge groups (+0.65 to +0.27, edge -0.51), and safe loops
barely move. In g170 the same move gets less (+0.99 to +0.77): the missing death signal
there is spread over more of the trunk.

**4. The loop feature is what silences them.** Hold the loop direction (safe-fit, as in the
eleventh run) at the doomed-edge level on the D chain:

| Hold | Death channels on doomed loops (share of doomed-edge level) | On safe loops | Doomed loop ownership |
|---|---|---|---|
| b40, none | 0.20 | 0.09 | +0.65 |
| b40, only at the input of block 38 | 0.19 | 0.09 | +0.68 |
| b40, blocks 12-24 | **0.59** | 0.24 | **-0.23** |
| g170, none | 0.10 | 0.09 | +0.99 |
| g170, only at the input of block 19 | 0.08 | 0.06 | +0.97 |
| g170, blocks 6-12 | **0.78** | 0.42 | **+0.26** |

Holding the loop feature in mid-trunk turns the death channels back on for doomed loops,
about twice as much as for safe loops. So the race information needed to fire them
is there once the loop signal is removed. Holding it only just before the late blocks does
nothing, so the silencing isn't a direct read of the loop direction by these channels. It
is set up in mid-trunk: the loop feature, present in blocks 12 to 24 (b40) or 6 to 12 (g170),
keeps the evidence of death from building on the chain, and the last blocks have nothing to
fire on.

The full mechanism, in the attacked net:

1. Mid-trunk builds a "closed loop" feature on the chain, in every net (tenth, eleventh runs).
2. In b40 and g170, that feature suppresses the build-up of "this group is dying" evidence on
   the loop's stones, most strongly when the loop is actually dying.
3. The last blocks hold a few dozen death-detector channels that write "dead" onto the
   ownership readout. On doomed loops they stay near silent (10 to 20% of their edge level),
   so the readout falls back on its default for a large connected chain, "alive".
4. tf3 has the loop feature (step 1) but not the suppression. Its doomed loops end on the
   dead side of the verdict axis (-0.06, twelfth run).

Caveats: channel-level analysis is on b40 and g170 only. tf3's blocks mix attention and
convolution, and I haven't decomposed them. Restoring uses the status label to choose the
edge reference, so it tests sufficiency of these channels, not whether the net could fire
them unaided. "Death channel" is defined by activation pattern on this set, not by any wider
test of what they respond to.

## Fourteenth run: where the loop blocks the death evidence

`goconcepts/gating.py`, eye-balanced set, after-pass state, fit on one half and tested on the
other. **Death axis** at each layer: from safe to doomed edge groups on the D chain's stones.
**Death evidence** on a chain: its position on that axis, with safe edge = 0 and doomed edge
= 1.

**1. The trajectory.** Death evidence on doomed loops by layer:

| Net | Early | Mid-trunk | Late | Output |
|---|---|---|---|---|
| b40 | 0.6 to 0.8 (blocks 0-4) | **-0.1 to -0.5** (blocks 10-22) | 1.12 at block 26, then down to 0.55 | 0.41 |
| g170 | 0.6 to 0.7 (blocks 0-3) | **-0.35 to -0.58** (blocks 6-11) | 0.37 at block 17 | 0.09 |

Through mid-trunk, doomed loops sit *below* safe edge groups on the death axis. Summed over
blocks, b40 writes 2.1 less death evidence on doomed loops than on doomed edge groups in
blocks 1 to 23 (g170: 2.8 in blocks 1 to 12). b40 then has a partial correction. At blocks
24 to 26 (block 26 is a global-pooling block) every ring jumps to the death side (doomed
loops 0.30 to 1.12, safe loops 0.00 to 0.92). But blocks 27 to 40 write 1.1 less death
evidence on loops than on edges, and the loops end at 0.41. That erosion is what leaves the
late death channels (thirteenth run) nearly silent.

**2. The loop feature is written on the "alive" side of the safety axis.** Cosine between
the loop direction (ring minus edge, safe positions) and the death axis, after-pass state:

| Net | Early | Mid-trunk (the window that matters) | Late |
|---|---|---|---|
| g170 | +0.06 to +0.15 | **-0.28 to -0.43** (blocks 6-11) | +0.16 to +0.44 (17-20) |
| b40 | +0.12 to +0.18 | **-0.17 to -0.36** (blocks 10-22) | +0.4 to +0.7 (26-40) |
| tf3 | -0.04 to -0.21 (blocks 0-4) | **+0.24 to +0.36** (blocks 6-11) | |

In mid-trunk, the vulnerable nets encode "this chain is a loop" partly *as* "this chain is
safe": the two share a subspace with opposite sign. So the loop feature itself counts as
evidence against death, and it pushes the chain off the death side while the race is being
read. tf3 flips the sign from block 6 on: in the adversarially trained net, a loop leans
toward "dangerous".

**3. When the loop is removed matters more than for how long (b40).** Hold the loop direction at
the doomed-edge level on the D chain, one block at a time:

| Hold (b40) | Doomed loop ownership | Safe loop |
|---|---|---|
| none | +0.65 | +0.93 |
| any single block from 12 to 23 | +0.14 to +0.30 | +0.80 to +0.86 |
| block 24 alone | +0.60 | +0.92 |
| blocks 12-19 | +0.06 | +0.73 |
| blocks 12-24 | -0.23 | +0.61 |

After a single hold at block 12, the loop signal regrows completely (back to 108% of its
normal ring-edge gap by block 26), yet the doomed loop still reads less alive (+0.28). The
death evidence that builds in the gap persists (+0.17 on the death axis at the output, mostly
through the last four blocks). So the gate is open or shut during blocks 12 to 23. Once
death evidence has been written on the chain, the regrown loop feature doesn't erase it.
By block 24 it's too late.

**g170 differs.** Single holds anywhere in 6 to 11 do little (+0.90 to +0.95), because the
loop signal regrows within four blocks (59% of its gap back by block 12) and suppresses
again. Only holding blocks 10 to 12 together (+0.51) or 6 to 12 (+0.26) works. g170
re-derives the loop feature redundantly. b40 builds it once and relies on it through the
window.

Putting it together, in the attacked net:

1. The loop feature is written mid-trunk in a direction that overlaps "safe" (cosine about
   -0.3 with the death axis). While present, it keeps death evidence from building on the
   chain (blocks 12 to 23 in b40).
2. A global-pooling block (26) partly corrects this, lifting every ring toward "dying".
3. Blocks 27 to 40 erode the death evidence on loops again, and the last blocks' death
   detectors stay near silent. The readout says "alive".
4. In tf3 the loop feature is present but points toward "dangerous" in the same mid-trunk
   layers, so it adds to the death evidence instead of blocking it.

Caveats: directions are mean differences, one per layer. The cosine comparison is between two
such directions, not a circuit. The "death evidence" scale is anchored on edge groups. tf3's
single-block holds weren't run (its cost here is about 15 s a forward pass). Its contrast
rests on the cosine and on the eleventh run's hold result.

## Fifteenth run: how the loop feature is built

`goconcepts/loopcircuit.py`, `goconcepts/closure_hold.py`, `scripts/closure_vs_loop_direction.py`.
The minimal pairs isolate closure: the cyclic and cut members differ at one point, the cut. Any
computation that depends on the ring being closed has to start there and spread.

**Input hygiene first.** Two KataGo inputs carry information about the whole position, and the
pairs expose both. (1) Under territory scoring, KataGo's feature code shifts the komi input by the
difference in stones each side has placed, so the two members of a pair differ in a *global*
input. For this run the cut member gets the cyclic member's global inputs. (2) The ladder planes
are computed chain-wide and differed far from the cut in some pairs. We kept only pairs with equal
liberty counts and no input difference beyond the stem's 5x5 reach of the cut: 131 of 400. With
both fixed, the closure signal at the stem is under 0.1 SD from 5 steps out and zero from 9, as it should be.
(The komi quirk also varies across the other position sets with the random board around them.
It isn't tied to status or geometry, but it is a nuisance input.)

**1. The signal spreads outward from the cut, locally.** Closure direction per layer = mean
(cyclic - cut) on ring stones at chain distance 5 or more, fit on half the pairs. Signal on the
other half, in standard deviations of the same projection across ring stones:

| Layer | g170, far ring stones (chain distance 8-12) | b40, same |
|---|---|---|
| 0 (stem) | 0.01 | 0.00 |
| 3 | 0.13 | 0.06 |
| 5 | 0.94 | 0.47 |
| 7 | 1.55 | 0.77 |
| 10 | 1.67 | 1.33 |
| peak | 1.69 (layer 9) | 1.76 (layer 19) |

The layer at which a ring stone first gets half its peak signal grows with its distance from the
cut: chain distance 1 to 3 at layers 2 to 4, 13 at layer 6 in g170 and layer 9 in b40. In g170 it
reaches every distance before the first global-pooling block (7). The wave is local propagation
through the convolutions, and it goes by board distance rather than along the chain. Across ring
stones, the signal falls 0.09 to 0.19 SD per point of board distance and 0 to 0.08 per step
along the chain. Stones whose path along the ring is at least 4 steps longer than their board
distance get the signal at layers 5 to 6 in g170, as fast as their board distance allows.

**2. Part of it goes through the enclosed region.** In the cyclic member, overwrite a region with
the cut member's states at every layer up to K, and measure how much closure signal is left at far
ring stones (chain and board distance 5 or more) at layer K:

| Region overwritten | g170, K = 6 | g170, K = 8 | b40, K = 6 | b40, K = 12 |
|---|---|---|---|---|
| enclosed interior | 0.62 | 0.61 | 0.57 | 0.65 |
| exterior band (2 points round the ring) | 0.94 | 0.95 | 0.90 | 0.95 |
| ring stones within 3 steps of the cut | 0.47 | 0.53 | 0.44 | 0.46 |
| interior and exterior band | 0.56 | 0.56 | 0.51 | 0.61 |

Blocking the interior removes 35 to 43% of the closure signal at far stones; blocking the outside
removes 5 to 10%. The rest arrives through the ring itself and nearby points. So the net partly
computes closure as a property of the *region*: a region sealed off from the outside, felt by
the ring stones that bound it.

**3. The closure signal is part of the loop direction exactly where it matters.** Cosine between
the closure direction and the ring-minus-edge loop direction of the earlier holds: about 0 at the
input layers (there the loop direction is mostly other geometry, such as distance from the board
edge), rising to 0.35 to 0.48 in g170's blocks 7 to 12 and 0.27 to 0.39 in b40's blocks 11 to 18.
In b40 the closure signal at far ring stones halves at the global-pooling block 26 (1.73 to 0.75),
the block that also lifts death evidence on loops.

**4. Pure closure drives the attacked net's error.** Holding the closure direction itself (from the
minimal pairs, so no edge geometry) at the doomed-edge level on the D chain, eye-balanced set,
held-out half:

| Hold | Doomed ring: ownership / capture | Safe ring | Doomed edge |
|---|---|---|---|
| b40, none | +0.65 / 0.04 | +0.93 | -0.51 |
| b40, closure direction, blocks 12-24 | **+0.25 / 0.15** | +0.70 | -0.52 |
| b40, random directions | +0.62 / 0.05 | +0.92 | -0.51 |
| g170, none | +0.99 / 0.00 | +0.99 | -0.26 |
| g170, closure direction, blocks 6-12 | +0.91 / 0.00 | +0.93 | -0.26 |

In b40, closure alone carries about half of what the full ring-minus-edge direction did (+0.65 to
-0.23), with random directions doing nothing. This answers the geometry confound raised in the
eleventh run, for the attacked net: a direction built only from "one stone closes the loop"
moves its verdict. In g170 it barely does. g170's "alive" leans more on the other ring-versus-edge
geometry, which fits its hopeless loops staying half alive (tenth run).

Caveats: the closure direction is one mean difference per layer. Only 131 pairs survive the
input filter, and 63 evaluation pairs have far ring stones. The region overwrite measures where
closure travels, not the channels that carry it.

## Sixteenth run: the channels of the closure wave, and tf3

`goconcepts/closure_channels.py`, `goconcepts/closure_hold.py` (wider windows),
`goconcepts/loopcircuit.py wave --net tf3`. Same 131 filtered minimal pairs as the fifteenth run.

**1. The wave has no small circuit.** Each block writes `fin(t)` into the trunk and `fin` is linear,
so a block's write along the closure direction splits exactly over its hidden channels. Ranked
over blocks 1 to 6 in g170 (1,536 channels), 66 channels carry half of the positive write and 211
carry 80%. Over blocks 1 to 12 it takes 128 and about 400, in both g170 and b40. There is no
handful of "closure neurons".

Some channels do specialise on the enclosed region. The interior is 3% of the board, yet it holds
20 to 32% of the cyclic-minus-cut difference in these channels. Several top channels in g170's
block 5 (78, 119, 215) put 83 to 99% of their difference inside the ring. They are the channel-level
form of the fifteenth run's finding that part of the wave crosses the interior.

**2. Knocking out the top channels does not stop it. The net rebuilds it.** In the cyclic member,
set the top-n channels to the cut member's values at every point, in every block up to K. Closure
signal left at far ring stones, and the ownership difference (cyclic minus cut, far ring stones,
ring owner's side), eval half:

| Net, channels patched | Signal at layer K | Signal later | Ownership difference |
|---|---|---|---|
| g170, top 128 in blocks 1-6 | 0.39 | 0.99 (layer 12) | 0.99 |
| g170, top 512 in blocks 1-12 | 0.29 | 0.74 (layer 16) | 0.95 |
| b40, top 512 in blocks 1-12 | 0.22 | 0.89 (layer 24) | 1.01 |
| random 512, either net | 0.79-0.82 | 0.93-0.94 | 0.99-1.02 |

Removing most of the closure signal early leaves the output untouched. The later blocks recompute
closure from what remains (the cut point's neighbourhood, the stones themselves). This matches the
fourteenth run, where the loop feature regrew after a mid-trunk hold. Closure is computed
redundantly, so channel ablation is the wrong tool for it. Direction holds on the trunk are the
right one.

**3. In g170 the closure direction is tied to "alive" in general, late.** The fifteenth run held it
over blocks 6 to 12 and got almost nothing. Wider windows, g170, eye-balanced set:

| Hold of the closure direction | Doomed ring | Safe ring | Doomed edge | Safe edge |
|---|---|---|---|---|
| none | +0.99 | +0.99 | -0.26 | +0.97 |
| blocks 6-12 | +0.91 | +0.93 | -0.26 | +0.91 |
| blocks 13-16 | +0.49 | +0.62 | -0.31 | +0.62 |
| blocks 1-16 | +0.45 | +0.65 | -0.31 | +0.62 |
| blocks 1-20 | -0.32 | -0.33 | -0.41 | -0.33 |
| random, 1-16 | +0.98 | +0.99 | -0.27 | +0.97 |

The whole effect sits in blocks 13 to 16, and it is not specific to rings: safe edge groups lose as
much as safe rings. By then the closure direction in g170 lies along a general life axis.
Compare b40 (fifteenth run, blocks 12-24): doomed rings -0.40, safe rings -0.23, safe edges only
-0.05. In the attacked net, closure is a ring-specific input to the verdict. In g170 it is not a
separate lever at all.

**4. tf3 computes closure at once, over the whole ring.** Same wave analysis on tf3 (conv plus
attention, 11 blocks). Closure signal on ring stones by chain distance from the cut, in SDs:

| Layer | Distance 1 | 4 | 8 | 12 |
|---|---|---|---|---|
| 0 (stem) | 0.51 | 0.29 | 0.11 | 0.00 |
| 1 | 1.58 | 1.70 | 1.13 | 0.77 |
| 2 | 0.95 | 1.66 | 1.65 | 1.58 |
| 5 | 0.81 | 1.58 | 1.94 | 2.27 |
| 11 | 0.66 | 0.97 | 1.17 | 1.41 |

Every ring stone reaches half its peak by block 1 or 2, against layers 2 to 9 in the convolutional
nets. From block 4 on the signal *grows* with distance from the cut: the regression slope is +0.09
to +0.18 per chain step and -0.07 to -0.12 per point of board distance, the opposite of g170 and
b40. The outside of the ring gets nothing (under 0.1 at every layer). tf3 builds closure as a
property of the whole chain, through attention, not as a wave from the cut. So tf3 does not lack
the closure concept. It has a cleaner one, and (eleventh run) it pushes loops toward death
instead of life.

Caveats: the channel patches set channels to the cut member's values, which also moves anything
else those channels carry. The tf3 closure direction is fit on far stones, which favours a signal
that is flat along the ring. The tf3 stem already differs up to 8 steps away, so its stem reach is
wider than the 5x5 filter assumed. The ownership difference on minimal pairs is a correct
judgement (a closed chain is one group, a cut ring is two), not the error itself.

## Seventeenth run: closure means "alive" in all three nets, tf3 included

`goconcepts/tf3_closure.py`, `goconcepts/closure_swap.py`, `scripts/closure_vs_death_cosine.py`.
The sixteenth run found that tf3 builds closure at once over the whole chain, and ended on the guess
that tf3 reads closure as a sign of danger. That guess is wrong.

**1. The closure direction points away from death in every net.** Closure direction per layer from
the 131 filtered minimal pairs (for tf3 about 260 forward passes). Cosine with the death axis
(safe edge to doomed edge, eye-balanced set, fit half), after-pass state:

| Net | Early blocks | Mid | Late |
|---|---|---|---|
| g170 | -0.03 to -0.33 (1-4) | -0.57 to -0.75 (6-12) | -0.79 to -0.87 (17-20) |
| b40 | +0.16 to -0.04 (1-4) | -0.32 to -0.62 (10-16) | -0.81 to -0.88 (26-40) |
| tf3 | -0.31 to -0.41 (1-2) | -0.62 to -0.71 (3-6) | -0.74 to -0.78 (7-11) |

Closure is a "this group lives" signal in all three. Along it, doomed edge groups sit about 2 SD
below safe ones by the last blocks in every net. The move state gives the same picture.

**2. The nets differ in how much death pulls rings down along it.** Doomed minus safe *rings* along
the closure direction, same units, last third of the net:

| Net | Doomed - safe edges | Doomed - safe rings | Doomed ring verdict |
|---|---|---|---|
| g170 | -1.8 to -2.2 SD | -0.06 to -0.36 SD | +0.99 (alive) |
| b40 | -1.2 to -2.3 SD | -0.55 to -0.77 SD | +0.66 |
| tf3 | -2.0 to -2.2 SD | -1.31 to -1.80 SD | -0.65 (dead) |

The order matches the verdicts. g170 keeps doomed and safe rings together on the life axis. tf3
pulls doomed rings most of the way down. The loop direction goes the same way: its cosine with
closure is 0.4 to 0.7 in the first half of every net. Near the end it falls to about 0 in g170
and tf3 and turns negative in b40 (-0.3 to -0.5) after its global-pooling block 26.

**3. Injected closure revives a doomed group in every net, tf3 included.** Swap hold: on the D
stones, set the closure component to the fit-half mean of the other geometry with the same status.
Edge groups get a ring's closure, rings get an edge group's. Eval half, 30 per cell:

| Net, blocks | Doomed edge: own / capture | Doomed ring: own / capture | Safe groups |
|---|---|---|---|
| g170, none | -0.25 / 0.48 | +0.99 / 0.00 | +0.97 to +0.99 |
| g170, swap 4-14 | **+0.96 / 0.00** | +0.87 / 0.01 | +0.99 |
| g170, random 4-14 | -0.16 / 0.47 | +0.99 / 0.00 | +0.97 to +0.99 |
| b40, none | -0.49 / 0.46 | +0.66 / 0.04 | +0.93 to +0.97 |
| b40, swap 7-29 | **+0.23 / 0.13** | +0.54 / 0.14 | +0.94 to +0.97 |
| b40, random 7-29 | -0.41 / 0.44 | +0.57 / 0.05 | +0.91 to +0.97 |
| tf3, none | -0.43 / 0.49 | -0.65 / 0.52 | +0.98 |
| tf3, swap 2-8 | **+0.68 / 0.08** | -0.74 / 0.67 | +0.97 to +0.99 |
| tf3, random 2-8 | -0.43 / 0.49 | -0.65 / 0.53 | +0.98 |

Giving a doomed edge group a doomed ring's closure makes all three nets read it as alive, and cuts
the capture move from about half the policy mass to 0.00 to 0.13. tf3 is as open to this as g170.
Taking closure away from rings does little anywhere (tf3 doomed rings -0.65 to -0.74, capture 0.52
to 0.67), which fits the sixteenth run: closure is rebuilt redundantly and rings have other support.

**What this means.** Adversarial training did not change what closure means to tf3. Closure is still
sufficient evidence of life. What tf3 added is death evidence on doomed rings strong enough to
override it: doomed rings sit lower on the closure axis, and some other ring-specific signal
finishes the job. It has to sit in the rest of the loop direction: in tf3 the whole direction
leans toward death (cosine +0.24 to +0.36, fourteenth run) while its closure part leans away. The level that doomed rings reach in tf3 would still make an edge group
read alive. So the fix is a counterweight tied to ring shape, not a corrected concept. It predicts
that a doomed group which carries closure but lacks the ring geometry tf3 learned to distrust,
for example an unusual enclosing shape, would still read as alive. That is a testable attack
direction on the hardened net.

Caveats: an edge group with a ring's closure is an off-distribution state, so the swap tests
sufficiency of the direction, not a position that can occur. 30 cases per cell. The windows cover
comparable fractions of each net (tf3 2-8 of 11, g170 4-14 of 20, b40 7-29 of 40), but tf3 is
shallower, so each held block carries more of its computation.

## Eighteenth run: tf3's fix holds on new loop shapes

`goconcepts/shapegen.py` (shape options added to `racegen.make_race`), `goconcepts/shape_eval.py`,
`goconcepts/tf3_counterweight.py`. The seventeenth run ended on a prediction: if tf3's extra death
evidence is tied to the shape of the training rings, a doomed loop with a different shape should
fool it. It does not.

**1. The test set.** 150 fresh races with exact labels, built like the eye-balanced set (5
liberties, doomed combo (2,1,3), safe (1,1,4), one real eye), 20 doomed and 10 safe per shape:
the standard thin ring (a within-set control), a ring two stones thick (44-60 stones), a ring with
an arm of 4-6 stones sticking out, a ring running along the first line, and a big ring (26-40
stones round a larger region). Doomed loops, ownership after a pass / capture mass on the correct
moves / fraction read alive:

| Shape | g170 | b40 | tf3 |
|---|---|---|---|
| standard | +0.99 / 0.00 / 100% | +0.66 / 0.03 / 100% | -0.58 / 0.46 / 20% |
| thick | +1.00 / 0.00 / 100% | +0.73 / 0.06 / 95% | **-0.88 / 0.50 / 5%** |
| arm | +0.99 / 0.00 / 100% | +0.64 / 0.04 / 100% | -0.60 / 0.37 / 20% |
| first line | +0.99 / 0.00 / 100% | +0.64 / 0.02 / 100% | -0.71 / 0.43 / 10% |
| big | +1.00 / 0.00 / 100% | +0.91 / 0.02 / 100% | -0.62 / 0.41 / 15% |

Safe loops read alive in every net and shape (+0.83 to +0.99). tf3 reads every new shape as dead
at least as well as the standard one. The prediction failed. b40 gets worse on big loops (+0.91),
which fits the earlier finding that loop size drives its error.

**2. The ring-shape direction carries only part of the counterweight.** Split tf3's loop direction
into its closure part and the rest (normalised), per layer. The rest leans toward death (cosine
+0.24 to +0.50 from block 3), and doomed and safe rings both sit 0.9 to 2.0 SD above edge groups
on it. Holding the rest at the doomed-edge level over blocks 3 to 9 (eval half, 30 per cell):

| | Doomed ring: own / capture | Safe ring | Doomed edge | Safe edge |
|---|---|---|---|---|
| none | -0.65 / 0.52 | +0.98 | -0.43 / 0.49 | +0.98 |
| hold the non-closure part, 3-9 | -0.54 / **0.29** | +0.99 | -0.43 / 0.49 | +0.96 |

Removing it costs tf3 some of its capture policy (0.52 to 0.29) but barely moves the ownership
verdict. So tf3's death evidence on doomed loops is not one ring-shape direction either.

**What this means.** The seventeenth run's reading, "a counterweight tied to ring shape", was too
narrow. tf3 reads doomed loops as dead across shapes it never saw in this form, and no single
geometric direction carries that. What all five shapes share is the race itself: an enclosed
opponent group with an eye, a few shared liberties, a surrounded chain. The likeliest account is
that adversarial training taught tf3 to read the capturing race around a loop, and that race
evidence outweighs closure's "alive". That is a real fix in the sense that matters for this attack
family. It leaves one clear gap to test: loops that die without a race.

Caveats: 20 doomed loops per shape, raw networks, no search. One seed. The holds use the same
directions as the seventeenth run and are one averaged direction per layer.

## Nineteenth run: tf3 also gets loops that die without a race

`goconcepts/hopeless_eval.py` (now resumable) on tf3. The eighteenth run proposed that tf3 reads the
capturing race around a loop, and predicted that loops dying without a race would fool it. The
hopeless set tests this: the enclosed group has two real eyes and shares no liberty with the loop,
so there is no race, and the loop dies on its 3 to 5 outside liberties whatever D does. 90 loops
and 90 edge-pocket controls. Ownership of the D chain after a pass (fraction read alive):

| Net | Hopeless loops | Hopeless edge groups | Enclosed group (truth: alive) |
|---|---|---|---|
| g170 | +0.15 (58%) | -0.92 (0%) | alive, -0.94 from D's side |
| b40 | -0.68 (3%) | -0.93 (0%) | alive, -0.99 |
| tf3 | **-0.97 (0%)** | -0.92 (0%) | alive, -1.00 |

tf3 reads every hopeless loop as dead, more firmly than the edge groups. The prediction failed
again. tf3's fix does not depend on a race, on the loop's shape (eighteenth run) or on one
direction we can hold (seventeenth and eighteenth runs). On every loop family we have built it
finds the death, and the injected-closure test shows it still treats closure itself as a sign of
life. The simplest reading left is that tf3 weighs the ordinary evidence of a surrounded chain
(outside liberties, the enclosing opponent stones) at full strength on loops, where the older
nets let closure mute it. We have not found the circuit that does this, and we have not found a
loop that fools tf3.

**A side result on the attacked net.** b40 reads hopeless loops as dead (-0.68, 3% alive) but
race loops as alive (+0.66, 100%). Its error is specific to loops in a capturing race, which is
exactly the situation the cyclic attack sets up. g170 fails on both. So the attacked net's blind
spot is narrower than g170's, and the attack found the part that remained.

Caveats: raw networks, no search. The hopeless loops are 90 positions from one generator.

## Twentieth run: an automated search for loops that fool tf3

`goconcepts/attack_search.py`, `goconcepts/race_ext.py`, `goconcepts/fortified_set.py`.

**0. tf3 is 40 times faster now.** Some tf3 activations are denormal floats, and MKL's fp32 matmul
runs at about 2% speed on them. `torch.set_flush_denormal(True)` in `katago_adapter.py` takes a
forward pass from 17 s to 0.4 s, with bit-identical outputs on the positions checked. Every
earlier tf3 run was slow for this reason alone.

**1. The search.** Start from doomed loops. Mutate 1 to 3 points at a time, mostly within 3 points
of the race (ring shape, liberties, surrounding stones), sometimes anywhere. Keep a mutant only if
it is still a closed race that D loses after a pass, solved exactly. Score = tf3's ownership of
the loop after D passes. Beam search per seed, width 3, 12 mutants per parent per step, 10 steps.

**2. The first search found real escapes, not tf3 errors.** On the original sets the search
flipped tf3 within 8 to 10 edits on all three seeds tried. In the first, the edits put D stones
against the opponent walls that seal the ring's outside liberties. Those walls were weak (2 or 3 liberties) in every
generated position: in all 119 valid seeds, some wall chain had fewer liberties than the ring.
The race solver plays only on ring and inner-group liberties, so it never lets D capture a wall
stone. A stricter solver that does (`race_ext.solve_ext`) showed the first flipped position is
savable by attacking a wall. tf3 was right and the label was wrong.

**3. Fortified walls.** We added opponent stones on wall liberties outside the race until every
wall chain had at least the ring's liberties plus 2, re-solved, and kept cases whose status did
not change (314 of 320). Readouts on the fortified eye-balanced set:

| Net | Doomed ring: own / capture / read alive | Doomed edge | Safe ring | Safe edge |
|---|---|---|---|---|
| g170 | +0.99 / 0.00 / 100% | -0.62 / 0.77 | +0.99 | +0.98 |
| b40 | +0.58 / 0.08 / 100% | -0.71 / 0.73 | +0.90 | +0.98 |
| tf3 | -0.86 / 0.62 / 1% | -0.77 / 0.85 | +0.99 | +0.99 |

The paper's main contrasts hold and get sharper: g170 and b40 still read every doomed ring alive,
and the ring-edge gap grows. tf3's failure rate drops from about 11% to 1%. Most of tf3's apparent
errors on the original set were positions where the weak wall gave D real chances. The g170 and
b40 readouts barely change, so the mechanism results do not rest on the weak walls.

**4. With strong walls, tf3 resists.** Search on 20 fortified seeds tf3 reads as dead (all below
-0.9), with mutants also required to keep every wall chain at ring liberties + 2. 17 seeds had
valid mutants:

| | Value |
|---|---|
| Seeds pushed above 0 (read alive) | 3 of 17 |
| Seeds pushed above +0.5 | 1 of 17 |
| Edits needed to cross 0 | 13 to 19 |
| Mean best score after 10 steps | -0.41 (from -0.93) |

The strongest one (bal181) reaches +0.66 after 20 changed points, and tf3's policy on the saving
capture falls from 0.79 to 0.002, so it would lose the ring in play. The edits change one ring stone
and one point next to the inner group, and add stones around the race. It is a real tf3 error under our label, but it
takes many edits, and the label assumes a wall with ring liberties + 2 cannot be attacked
profitably. A search-based engine check is the right confirmation.

**What this means.** tf3's fix is robust to small, local changes: 14 of 17 doomed loops stayed dead
through 10 rounds of selection. The cheap way to "fool" it was to give D a real escape, which it
read correctly. Separately, our generated races had weak walls throughout. The ring results stand,
but the doomed labels on the original sets assume the walls hold.

## Twenty-first run: do the flips survive search?

`goconcepts/mcts.py`. KataGo's C++ engine is not available here (no build, and tf3 is a PyTorch
checkpoint), so this is a small PUCT search over tf3 itself: 800 visits, all legal moves, batches
of 8 with virtual loss, Japanese rules. It asks whether search corrects the raw net on the three
positions the twentieth run flipped. It is tf3 judging with search, not an independent referee.

**A point of method.** The first version used win/loss only. The random board around our races
leaves D lost overall (root value -1.0 everywhere), so that search had nothing to prefer and
followed its priors. KataGo adds a score term to its utility. With one (0.3 x atan of the lead
relative to the root's, scale 14 points) the search behaves sensibly on the unflipped seeds, so the
numbers below use it. An early run also kept only the top 40 moves per node, which cut the saving
capture in one position. Both earlier result files are kept.

**Results.** Ownership of the loop from D's side, raw net against 800-visit search:

| Position | After D passes: raw / search | D to move: visit share on the saving capture | D to move: search ownership |
|---|---|---|---|
| bal12 seed | -0.92 / -0.90 | 0.92 | +0.82 |
| bal12 flipped | **+0.38 / -0.78** | 0.64 | +0.27 |
| bal161 seed | -0.93 / -0.93 | 0.99 | +0.96 |
| bal161 flipped | **+0.43 / -0.74** | 0.93 | +0.91 |
| bal181 seed | -0.92 / -0.92 | 0.95 | +0.87 |
| bal181 flipped | **+0.66 / -0.76** | **0.00** | -0.31 |

1. **Search corrects the misread in all three.** After D passes, 800 visits take the loop from
   "alive" (+0.38 to +0.66) to dead (-0.74 to -0.78). In two of the three the opponent's most
   visited move is the one it plays on the unflipped seed. So the search agrees with our label that these loops die after a pass. The raw-net
   errors the attack search found do not survive 800 visits.
2. **One failure remains, of a different kind.** In bal181, with D to move, the search never visits
   either saving capture (raw priors 0.0016 and 0.0006) and plays elsewhere. It no longer thinks
   the loop is safe (ownership -0.31). It thinks the loop is dying and does not see the save. If
   our solver is right that the capture saves it, tf3 with 800 visits still loses that loop. In
   bal12 the search finds the capture less often than on the seed (0.64 against 0.92) but still
   plays it most.

**What this means.** For tf3, the misjudgements we could construct are shallow: the raw net gets
them wrong and a modest search gets them right. That differs from the nets in the paper, where
FAR's attack beat b40 at millions of visits. The remaining weakness is a missing candidate move,
not a wrong belief about loops.

Caveats: three positions. A simplified search (no symmetry averaging, fixed exploration constant,
our own score utility). The engine being tested is also the judge, so this corroborates the label
but does not prove it.

## Twenty-second run: the blind spot survives search in g170 and b40

`goconcepts/mcts.py --set` (the search now runs on the `.bin.gz` nets too; `kgbin_adapter.py`
returns the score lead). The same 800-visit search as the twenty-first run, on the fortified
eye-balanced set: 12 doomed rings, 4 doomed edge groups and 4 safe rings, picked at random with a
fixed seed. Each net searches with itself. tf3 was run on the 12 doomed rings only.

| Net | Doomed ring after a pass: raw / search | Read alive by search | Doomed ring, D to move: visits on the saving capture | Doomed edge: after a pass / capture visits |
|---|---|---|---|---|
| g170 | +0.99 / +0.98 | 12 of 12 | 0.00 | -0.86 / 0.99 |
| b40 | +0.55 / **+0.78** | 12 of 12 | 0.03 | -0.93 / 1.00 |
| tf3 | -0.90 / -0.85 | 0 of 12 | 0.87 | not run |

1. **Search does not fix it.** With 800 visits g170 and b40 still read every doomed ring as alive,
   and b40's reading gets worse (+0.55 to +0.78). With D to move they put almost no visits on the
   capture that saves the ring (0.00 and 0.03). On edge groups the same nets with the same search
   put 99 to 100% of visits on it. The search works. It is the loop that blinds it.
2. **Both sides are blind.** After D passes it is the opponent's turn, and the search is the same
   net playing the attacker. It does not find the kill either: the ring stays "alive" in its own
   search. This is the situation in FAR's games, where the victim does not defend a loop its own
   search believes is safe.
3. **tf3 is right before and after search**, and finds the saving capture with 87% of visits.

Why search fails here: the value and ownership heads say "alive" at every node the search visits
near the loop, and the policy gives the capture little prior (0.00 and 0.08), so there is nothing for the search
to find within 800 visits. This is the mechanism of sections 6.1 to 6.5 seen from outside: the
error is in the evaluation, not in a missing variation.

Caveats: 12 rings per net, 800 visits against the millions in FAR's strongest games, our own
simplified search and score utility. Safe rings stay alive under search in both nets (+0.92 and
+0.99), so the search is not simply pushing everything one way.

## Twenty-third run: holding the loop direction during search

`goconcepts/mcts.py --set --hold loop` (and `kgbin_adapter.forward_hold`). The causal result of the
eleventh run was on raw readouts. Here the same intervention runs inside the search: at every node
b40 evaluates, the loop direction (safe ring minus safe edge, blocks 12 to 24, fit on the fit half
of the original eye-balanced set) is held at the doomed-edge level on the ring's stones still on
the board. The direction and target switch with the side to move, as in the earlier holds. Same 20
positions and 800 visits as the twenty-second run. Control: random directions, same blocks.

| b40, 800 visits | Doomed ring after a pass | Read alive | Visits on the saving capture | Safe ring after a pass | Doomed edge after a pass / capture visits |
|---|---|---|---|---|---|
| no hold | +0.78 | 12 of 12 | 0.03 | +0.92 | -0.93 / 1.00 |
| hold the loop direction | **-0.24** | **1 of 12** | **0.23** | +0.38 | -0.75 / 0.99 |
| hold random directions | +0.75 | 12 of 12 | 0.02 | +0.92 | -0.94 / 0.99 |

1. **The search's verdict flips.** With the loop direction held, b40's search reads 11 of 12
   doomed rings as dead after a pass (mean -0.24, from +0.78). Random directions change nothing.
   So the one direction that carried the raw misjudgement also carries it under search.
2. **The saving capture comes back, partly.** Visits on it rise from 0.03 to 0.23. In 2 of 12
   positions it becomes the majority choice (0.57 and 0.78), and in 3 it passes a quarter of the
   visits. No position passed 0.06 without the hold. b40 now believes the loop is in danger more
   often than it finds the move that saves it.
3. **The cost.** Safe rings drop from +0.92 to +0.38 after a pass, and 1 of 4 reads as dead. The
   hold removes the loop's protection whether or not the loop deserves it, as in the raw holds.
   Edge groups are untouched.
4. Nine of the 12 rings are outside the half the direction was fit on. On those nine the numbers
   are the same (-0.18 after a pass, 0.27 on the capture, against +0.73 and 0.02 for random).

**What this means.** The mechanism found on raw readouts is the one that blinds the search. One
direction per block, held on about 20 stones, turns a search that reads every doomed loop as
alive into one that reads nearly all of them as dead. This is a causal link from an internal
feature to play under search. It is not a fix: it damages safe loops and only partly restores the
right move.

Caveats: 12 rings, 800 visits, our own search. The hold's target is the doomed-edge level for every
position, so it does not use any position's true status.

## Next steps, in order

1. **Write it up.** The chain from real-game failure to mechanism is complete enough for a
   paper or post: shared blind spot, safety concept, attacked net, loop feature, its
   entanglement with "safe", tf3's sign flip.
2. **Independent engine check.** Run the fortified doomed rings and the three tf3 flips through
   KataGo's own engine on a machine that has it, as the referee our self-judging search is not.
   (The paper's abstract, introduction and contributions were rewritten on 2026-10-02 to cover
   sections 6.6 to 6.12.)
3. **Search.** Rerun the key positions through KataGo's engine with a few hundred visits
   (needs GPU or the C++ engine on your machine), to show the misjudgement survives search.
4. **Human side (stage four).** Human games with big loops are rare. Use the human attack
   games (JBXKata005, LZ 4096) to test whether amateurs misjudge the same loops.
5. Scale the minimal pairs to 2,000 and add a 1-liberty (atari) condition.
6. Non-linear probes (small MLP over a 3x3 neighbourhood) to test whether the decay with
   depth is a readout artefact or real loss of information.
