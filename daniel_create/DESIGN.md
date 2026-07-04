# Design notes: grid generation and image labeling optimizations

This document explains the optimizations that let `daniel_create` produce and score a
full $t_{\text{start}} \times t_{\text{end}}$ ChordEdit grid for a large image set at a fraction of
the naive cost. The work is split into two independent stages that communicate only
through cell images written to disk: `generate_grid.py` synthesizes every grid cell,
and `label_grid.py` scores the cells that already exist. Keeping the stages separate
is itself an optimization: generation never pays to load the CLIP tower, labeling never
pays to load the diffusion pipeline, and either stage can be re-run, resumed, or
re-sharded without redoing the other. Every optimization below is exact in the sense
that it reproduces the reference `ChordEditPipeline.__call__` output bit-for-bit (or to
within documented floating-point tolerances), so the speedups never change the science.

## 1. The problem and the naive baseline

A grid ablation sweeps two timesteps independently: the transport start time
$t_{\text{start}}$ and the cleanup end time $t_{\text{end}}$. With $N$ sampled values on each axis
(here $N = 11$, i.e. $\{0.0, 0.1, \dots, 1.0\}$), each source image produces an $N \times N = 121$
cell grid. The reference implementation (`scripts/run_grid_ablation.py`) calls the entire
editing pipeline once per cell. Each of those $N^2$ calls repeats the same expensive
preamble: it encodes the source image into the VAE latent space, encodes the source and
target prompts, draws the editing noise, runs the transport step, runs the cleanup step,
and finally decodes the latent back to pixels. Because the preamble does not depend on the
particular $(t_{\text{start}}, t_{\text{end}})$ cell, the naive scheme performs on the order of $N^2$
redundant VAE encodes and $N^2$ redundant prompt encodes, and — most importantly — it
recomputes the transport (the single most expensive operation) for every one of the
$N^2$ cells even though transport only depends on $t_{\text{start}}$. This redundancy is the
target of the generation optimizations.

## 2. Factorizing the grid (`pipeline_ops.run_factorized_grid`)

The central observation is that for a single-step edit ($\texttt{n\_steps} = 1$) the ChordEdit
computation factorizes cleanly along the two grid axes. With one step the edit reduces to
three stages: a transport that moves the source latent $x_{\text{src}}$ along an estimated
edit direction, a cleanup that projects the transported latent back toward the clean-image
manifold at time $t_{\text{end}}$, and a decode. Concretely,
$$
x_{\text{transport}} = x_{\text{src}} + s \cdot \hat{u}(x_{\text{src}}, t_{\text{start}}, \delta),
\qquad
x_0 = \operatorname{pred\_x0}(x_{\text{transport}}, t_{\text{end}}),
\qquad
\text{image} = \operatorname{VAE\_decode}(x_0),
$$
where $s$ is the step scale and $\delta$ is the (fixed) `t_delta`. The key structural fact is
that $t_{\text{start}}$ enters **only** through the transport $\hat{u}$ and $t_{\text{end}}$ enters
**only** through the cleanup $\operatorname{pred\_x0}$. Therefore the shared preamble (VAE encode,
prompt encodes, noise draw) can be run exactly once per image; the transport can be run
once per unique $t_{\text{start}}$, i.e. $N$ times instead of $N^2$; and only the cleanup and decode
must run per cell. On the $N \times N$ grid this turns $N^2$ full pipelines into a single encode,
$N$ transports, $N^2$ cleanups, and $N^2$ decodes. Since a transport issues several batched
UNet forwards and is by far the dominant cost, collapsing $N^2$ transports down to $N$ is the
biggest single win — roughly an $N$-fold reduction in transport work (an order of magnitude at
$N = 11$). The factorization is guarded by `settings.require_factorizable_config`, which refuses
to run unless $\texttt{n\_steps} = 1$, because the clean separation of $t_{\text{start}}$ and
$t_{\text{end}}$ only holds for the single-step edit.

## 3. The $\delta = 0$ transport fast path (`pipeline_ops._u_estimate_delta0`)

The transport direction $\hat{u}$ is itself expensive: the reference estimator
`_u_estimate_default` issues **four** batched UNet forwards. It evaluates the predicted clean
latent under both the source and target conditionings, at two noise levels — the anchor
renoised at $t_{\text{start}}$ and the anchor renoised at $t_{\text{start}} - \delta$ — and then blends
the two resulting edit directions. Writing $dv_s$ for the source-vs-target direction computed at
$t_{\text{start}}$ and $dv_{s_0}$ for the direction computed at $t_{\text{start}} - \delta$, the reference
blends them as
$$
\hat{u} = \frac{\delta \cdot dv_{s} + t_{\text{start}} \cdot dv_{s_0}}{t_{\text{start}} + \delta}.
$$
When $\delta = 0$ this collapses exactly. The second noise level equals the first, so the renoised
anchor $z_{\text{prev}}$ is identical to $z_s$; the last two forwards therefore duplicate the first
two, giving $dv_{s_0} = dv_s$. Substituting into the blend yields
$$
\hat{u} = \frac{0 \cdot dv_s + t_{\text{start}} \cdot dv_s}{t_{\text{start}} + 0} = dv_s,
$$
so the entire estimate reduces to the two unique forwards that produce $dv_s$. The fast path runs
exactly those two forwards — batched as $[z_s(\text{src}), z_s(\text{edit})]$ with a single shared
timestep and broadcasted $\alpha$/$\sigma$ coefficients — cutting transport cost in half while
remaining bit-for-bit identical to the four-forward reference at $\delta = 0$. This shortcut is only
valid in the default edit mode; the symmetric (`sym`) mode uses a different second timestep
($1 - t_{\text{start}}$ rather than $t_{\text{start}} - \delta$), so it has no equivalent collapse and is
routed to the reference path. Note also that no $t = 0$ shortcut is applied: even though the noising
term is nominally the identity at $t = 0$, the UNet still returns a non-trivial update there, so
skipping the forward would be numerically wrong; the fast path only exploits the exact algebraic
duplication caused by $\delta = 0$.

## 4. Batched cleanup and per-cell decode (`pipeline_ops._cleanup_decode_row`)

After transport, the grid still needs a cleanup and a decode per cell. For a fixed
$t_{\text{start}}$, a whole row of the grid shares the transported latent, the target conditioning, and
the noise — only the cleanup timestep $t_{\text{end}}$ varies. The cleanup (`_pred_x0`) is therefore
batched: the shared latent is repeated $N$ times and the $N$ distinct $t_{\text{end}}$ timesteps are
concatenated into a single UNet forward of batch size $N$, which is measurably faster than $N$
single-item forwards because it amortizes kernel-launch and memory-bandwidth overhead. The VAE
decode, by contrast, is deliberately **not** batched. Empirically the SD VAE already saturates the
GPU at batch size one (~168 ms/image); batching the decode actually regresses throughput
(~195–220 ms/image), which would erase the cleanup win. So the decode is kept per-cell, and only the
final device-to-host transfer and PIL conversion (`_tensor_to_pil`) are done once on the concatenated
batch, collapsing $N$ separate GPU→CPU copies into one. This mixed strategy — batch what benefits from
batching, keep serial what does not — is driven by measured throughput on this specific VAE rather than
by a blanket "batch everything" rule, and its output matches the reference per-cell loop to within the
fp16 decode tolerance ($\le 2/255$ per channel).

## 5. Numerical throughput settings

Because every forward in the grid has a constant input shape (the latent resolution and batch
structure are fixed across cells), the generation stage enables TF32 matmuls
(`torch.backends.cuda.matmul.allow_tf32`), TF32 in cuDNN, and cuDNN autotuning
(`torch.backends.cudnn.benchmark = True`). Autotuning pays off precisely because shapes never change,
so cuDNN can select an optimal kernel once and reuse it for the entire run; TF32 accelerates the fp32
UNet forwards and the VAE decode (the dominant costs) with numerically negligible impact on the edits.

## 6. Labeling optimizations (`label_grid.py`)

The labeling stage scores every generated cell with two metrics adapted from PnPInversion:
whole-image PSNR between the source and the edit, and a mask-restricted CLIP similarity between the
edited region and the target prompt. Two things make this cheap. First, the metric math is inlined into
the same `chordedit` environment rather than shelling out to a second env, so cells are scored in-process
immediately after loading. Second, both metrics are batched across the whole grid. PSNR is computed in a
single GPU operation: all $N^2$ predictions are stacked against the one source image and the per-image
mean squared error is taken over the channel and pixel axes, after which
$$
\text{PSNR} = -10 \cdot \log_{10}\!\big(\operatorname{MSE}\big)
$$
is applied elementwise (with $\texttt{data\_range} = 1$). This is mathematically identical to invoking
torchmetrics' `PeakSignalNoiseRatio` once per cell, but replaces $N^2$ metric calls with one batched
reduction. The CLIP metric masks every image to the edit region up front and then runs the CLIP image
tower in chunks (batch size 64), encoding the single target prompt once per chunk instead of once per
image; the score is the standard $100 \cdot \cos(\text{image\_embed}, \text{text\_embed})$ on
$L^2$-normalized embeddings, reproducing torchmetrics' `CLIPScore`. Chunking the image tower turns
$N^2$ forwards into $\lceil N^2 / 64 \rceil$ forwards while keeping the arithmetic unchanged.

## 7. Resumability and multi-GPU sharding

Both stages are designed to be interrupted and resumed without wasted work, and to scale across GPUs.
Generation skips any sample whose full set of cell files already exists on disk, so a re-run only fills
in genuinely missing cells. Labeling reads back every `result*.csv` in the output directory and skips
any sample already scored, and it flushes the CSV after each sample so a partial run is never lost.
Parallelism is achieved by round-robin sharding the sorted sample list: with $P$ GPUs, shard $i$ handles
samples $i, i+P, i+2P, \dots$, giving each worker a disjoint and balanced slice with no coordination or
locking. Each labeling shard writes its own `result_shard<NN>.csv` to avoid concurrent-write races, and
the shard CSVs are concatenated into a single `result.csv` (header once) after all workers finish. The
shell wrappers treat the GPU set as a required input — `GPUS` never silently defaults to grabbing several
devices; when it is unset the scripts fall back to the minimum requirement of a single GPU (index 0) and
say so — which makes the resource footprint of a run explicit and intentional.
