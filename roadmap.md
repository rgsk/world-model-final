# Roadmap

A playable world model: press an arrow key, and a neural network draws the next
frame of the game. A clean rewrite of ../../ml/world_model in the style of
../llm-final and ../asr: .pyn files with byname records, one module plus its
tests per step.

What the earlier project showed:
- The Genie pipeline (FSQ video tokenizer, unsupervised latent actions, MaskGIT
  dynamics) gave sharp frames that ignored the arrow keys, even after
  retraining on measured motion labels.
- A pixel model trained with MSE followed the keys but blurred, because it
  averages every possible next frame.
- A warp model (predict optical flow, shift the frame) was sharp and 96-99.8%
  correct on Zelda, but can only move pixels already on screen: nothing new
  appears at the edge during a scroll, and enemies or doors can't change.

The plan keeps what worked (labeled actions, a synthetic world where control is
checkable exactly, measuring the direction of predicted motion) and predicts
the next frame with flow matching, as in ../diffusion-final, conditioned on the
last few frames and the action. Flow matching draws one sharp frame instead of
an average, and can draw content that wasn't on screen. This is how DIAMOND
(Atari, CS:GO) and GameNGen (Doom) make playable games.

## Done
1. Setup. llm-final's project layout with byname and pytest, plus paths to
   the earlier project's Zelda frames (72,410 at 128x128).
2. Synthetic world. A red square on a green 64x64 field, moved 6 px by 5
   actions (noop, up, down, left, right) and stopped by the walls. Clips are
   generated on the fly, and sprite_center measures where a frame draws the
   square, which is how a model's control will be checked.
3. Flow-matching next-frame model on the synthetic world. diffusion-final's
   UNet (3.15M params) with the 2 previous frames stacked onto x_t as
   channels and the action added to the time embedding. Every eval draws the
   next frame for 200 fixed states per action: 100% control from step 500 to
   4000, square within 0.01 px, pixels off by 0.0054 on average (under one
   0-255 level). Peak lr 2e-3 reached 100% then collapsed; 5e-4 is stable.
   15 minutes on the 4060. src/show_world.pyn draws a run's frames next to
   the truth.

## Next
4. Zelda data. Frames from the h5 file, cut into short clips, with
   self-supervised motion labels (scroll direction from FFT phase correlation,
   as in the earlier project).
5. Baseline. The warp model, to beat on Zelda.
6. Flow matching on Zelda. Compare with the warp baseline on control accuracy,
   sharpness, and what appears at the edge during a scroll.
7. Rollouts. Many steps in a row, each from the model's own previous frames;
   find where they drift and fix it (noise on the context frames, as DIAMOND
   does).
8. Play it. Arrow keys drive the model in real time.
