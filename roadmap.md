# Roadmap

A playable world model: press an arrow key, and a neural network draws the next
frame of the game. A clean rewrite of ../../ml/world_model in the style of
../llm-final and ../asr: .pyn files with byname records, one module plus its
tests per step.

The longer aim is robot-arm video, which comes without the arm's actions, so the
actions have to be learned from the frames alone. Games with recorded keys (and
the synthetic world, where every move is known) serve as the answer key that
scores the learned actions.

What the earlier project showed:
- The Genie pipeline (FSQ video tokenizer, unsupervised latent actions, MaskGIT
  dynamics) gave sharp frames that ignored the arrow keys, even after
  retraining on measured motion labels.
- A pixel model trained with MSE followed the keys but blurred, because it
  averages every possible next frame.
- A warp model (predict optical flow, shift the frame) was sharp and 96-99.8%
  correct on Zelda, but can only move pixels already on screen: nothing new
  appears at the edge during a scroll, and enemies or doors can't change.

The plan keeps what worked (a synthetic world where control is checkable
exactly, measuring the direction of predicted motion) and predicts the next
frame with flow matching, as in ../diffusion-final, conditioned on the last few
frames and the action. Flow matching draws one sharp frame instead of an
average, and can draw content that wasn't on screen. This is how DIAMOND
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
4. Play the synthetic world. src/play_world.pyn drives the model with the
   arrow keys, each frame drawn from its own previous frames, with the true
   world alongside. Over 8 games of 300 random presses the square never
   drifted more than 0.06 px, so this world is too easy to study drift.
5. Actions learned from frames alone, stage 1 (src/latent_action.pyn,
   src/train_latent.pyn). A guesser sums up the change between two frames as a
   point z; a small predictor draws the second frame from the first and z in
   one step, by pixel error. For 500 steps the predictor reads z itself, which
   spreads the moves into clumps; then k-means++ places a codebook of 8 codes
   on the clumps and the nearest code replaces z (VQ). No true action is shown.
   Its codes stand for the true moves 98.1% of the time: one code per move,
   spare codes for moves cut short by a wall. 2.5 minutes. What failed first:
   training the guesser through the flow model (every move on one code),
   hard top-score picks (stuck at 3 codes), restarting unused codes by cloning
   or on the worst transition, and K = 5 codes, which left no slack and merged
   two moves in one of two runs.
6. Stage 2 (src/train_coded.pyn). The flow model of step 3, trained on the
   frozen guesser's codes instead of the keys; the control check presses each
   key as the code of its move. It matches the true-keys run: 100% control
   from step 500, best pixel error 0.0052 against 0.0054.

Parked on branch zelda-scroll-labels: Zelda clips labeled by scroll direction
(FFT phase correlation). The labels follow the camera, not Link, and on plain
floors the measurement locks onto Link and gives the reversed direction, so they
serve to score learned actions rather than to train on.

## Next
Not decided yet; candidates, closest to the robot aim first.
7. A harder synthetic world: a second object that drifts on its own, so not
   all motion comes from the player. The guesser must learn to ignore motion
   it doesn't cause, and the codes can still be scored exactly.
8. Learned actions on Zelda, scored against the scroll labels where those are
   clean.
9. Rollouts. Many steps in a row, each from the model's own previous frames,
   in a world hard enough to drift; fix it (noise on the context frames, as
   DIAMOND does).
10. Play Zelda with the arrow keys in real time.
