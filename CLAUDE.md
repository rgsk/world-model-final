# world-model-final

A playable world model: press an arrow key, and a neural net draws the next game
frame (Zelda). The next frame comes from flow matching, conditioned on past frames
plus the action. `roadmap.md` has the plan and what's done.

## Working
- Go one step at a time. Don't start the next step until the user asks.
- Commit only when asked, straight to `main`. The message is one short subject line
  (e.g. "synthetic world") with no body, only the `Co-Authored-By` trailer.

## Checks, after every edit
- `uv run pytest -q`
- `uv run byname format <changed files>`
- `uv run byname tool basedpyright src tests`
- `uv run annotation-check <changed .pyn files>` (from ../ml-tools)
  - `name is Any` (or `unknown`, `partially unknown`): the name shows white in the editor. Annotate it
    where it's first bound. A line with `# any:ok` is skipped.
  - `T is Unknown to the checker`: the annotation itself is untyped. Fix the type, not the annotation.
  - `T is inferred anyway`: a redundant annotation. Remove it, unless it's marked `# any:explicit`.
    A name assigned again in its scope is never reported, because there the annotation guards those
    assignments.
- Don't use ruff check.

## Running
- Plain Python can't import .pyn. Use `uv run byname run src/x.pyn`; notebooks need `%load_ext byname`.
- Run long jobs in the background and watch with `tail -F <log> | grep --line-buffered`.
  Stop old watchers before starting a new run on the same log, or every event shows up twice.
- Don't benchmark while training uses the GPU (an 8 GB RTX 4060). Sampling checks can
  run on the CPU with `CUDA_VISIBLE_DEVICES=`.
- `pkill -f <pattern>` kills its own shell, so use `pgrep -f '[t]rain_world'`.
- `RUN_NAME` names `artifacts/logs/<run>.log` and `artifacts/checkpoints/<run>.pt`.
  `show_world.pyn <run>` writes `artifacts/samples/<run>.png`.

## Code style
- Comments are short, technical and precisely true. Every file starts with a header
  comment. No mid-sentence colons; write "because" or "so".
- Every forward starts with an in/out shape header (`# x: [B, C, H, W]`). Add a shape
  comment only where the shape changes. Avoid heavy composition.
- Naming:
  - one meaning per name (`world.step` is the world's rule; `step` alone is a training step)
  - functions named by method; no `get_`
  - always use the `name=` shorthand
  - inputs are positional, settings keyword-only
  - shapes are destructured with every dimension named
- Tests read top to bottom like a walkthrough: named pieces, no magic numbers. Annotate
  fixture parameters, and annotate the return type of fixtures that build records.
- byname:
  - records for values that travel together (configs, batches, results); tuples only
    when unpacked on the spot
  - `rec._replace(...)` to change fields; `model(**batch)` passes a record by name
  - `record(T, dict)` when loading a checkpoint config; `_asdict()` when saving
- Open, closed or keyword arguments?
  - **Open:** a part of a model reads a slice of the model's config (`Block(cfg)`).
  - **Closed sub-record:** a separate concern with one consumer that uses all of it
    (`Schedule` for `lr_at`).
  - **Keyword arguments:** a few settings with no natural group.
- byname feedback goes to `../byname/feedbacks/world_model_final.md`: crisp, with code examples.
