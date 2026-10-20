# Existing assets and tools

- `web/vendor` is reused from the user's `q1_wheel_urdf` project. Three.js is MIT licensed: https://github.com/mrdoob/three.js . URDFLoader is Apache-2.0 licensed: https://github.com/gkjohnson/urdf-loaders . Preserve upstream license obligations when redistributing.
- `scripts/isaac_env.sh` and `scripts/reinforcement_learning/rsl_rl/runtime_compat.py` are adapted/reused from the user's wheel project.
- `meshes/q1_inertia`, `urdf/q1_inertia/quanta_robot_legged.urdf`, and `assets/upstream/bu10_robot_assets` are user-provided robot CAD. No redistribution license is asserted here.
- Reference video: https://youtu.be/KKhF516rSyo . Downloaded first seven seconds for the requested motion analysis; video artifacts are excluded from Git by default.
- MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker . Model is downloaded by the preparation script and excluded from Git.
