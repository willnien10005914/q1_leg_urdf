# Native PPO playback and recording

Run from `~/projects/q1_leg_urdf`.

## Isaac Sim window

```bash
cd ~/projects/q1_leg_urdf
bash play_native.sh --skill sidestep
# Or standing:
bash play_native.sh --skill stand
```

This runs the selected native PPO checkpoint in a single PhysX environment, with a camera aimed at the robot. Episodes repeat in the GUI. Close the window or press Ctrl+C to exit. Default seed is 42.

## Web and live PhysX

```bash
cd ~/projects/q1_leg_urdf
bash start_native_web.sh
```

Open http://127.0.0.1:8876/web/ and select **閃身步** and **即時 PPO / Isaac PhysX**. Use the replay button to restart the finite episode. **PPO 物理評估記錄** plays the saved evaluated trajectory; **參考動作 / 關節映射** plays the MediaPipe reference.

The launcher starts the HTTP server and the headless Isaac Sim live worker as separate background services. Stop with:

```bash
systemctl --user stop q1-leg-live q1-leg-web
```

## Record and upload

```bash
cd ~/projects/q1_leg_urdf
bash record_sidestep.sh
# Explicitly upload the newly recorded MP4 to GoFile:
UPLOAD_GOFILE=1 bash record_sidestep.sh
```

Default output: `docs/results/q1_leg_sidestep_6144_8000.mp4`, H.264, 1280×720, 50 FPS. The adjacent JSON records seed, checkpoint SHA-256, finite episode duration, survival, and lateral excursion. The script checks that the full episode survived before uploading. GoFile upload responses are saved beside the video; share the `downloadPage` URL.

## Completed 6144 / 8000 run

Both standing and sidestep completed 8000 iterations with 6144 parallel environments. Selection excluded historical runs and evaluated saved snapshots from this run. The selected standing checkpoint is `model_900.pt`, with 32/32 surviving fresh-seed confirmation episodes; the selected sidestep checkpoint is `model_7800.pt`, with 31/32 surviving fresh-seed episodes. Both passed the configured simulation gates. Selected files are promoted to `checkpoints/isaac/<skill>/latest.pt` with matching evaluation and contract files.

This is native CAD simulation evaluation, not validation on physical hardware. A successful demonstration episode does not mean every initial condition succeeds. Recording attempts and seed are retained locally; seed 50000 failed in a single-environment attempt, while seed 42 completed the finite sidestep episode.
