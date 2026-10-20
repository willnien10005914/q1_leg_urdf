#!/usr/bin/env python3
"""MediaPipe VIDEO -> confidence-gated world landmarks -> native-axis Q1 IK.

Only [0, 7) seconds are read. Image hip displacement supplies an approximate
root track (monocular scale/camera ambiguity is recorded in the artifact).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/leg_humanoid_lab"))
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from leg_humanoid_lab.robot import ROOT, URDF, JOINTS, DEFAULT, LOWER, UPPER, HEIGHT, fk, grounded_height

EDGES = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]
KEY = [11,12,13,14,15,16,23,24,25,26,27,28]
def unit(x):
    return x / max(np.linalg.norm(x), 1e-8)

def features(q):
    t = fk(q)
    out = [unit(t["center_torso_link"][:3,:3] @ [0,0,1])]
    for side in ("left", "right"):
        points = [t[f"{side}_{part}_link"][:3,3] for part in ("hip_yaw", "ankle_pitch", "foot", "shoulder_yaw", "wrist_yaw", "gripper")]
        out += [unit(points[1]-points[0]), unit(points[2]-points[1]), unit(points[4]-points[3]), unit(points[5]-points[4])]
    return np.array(out).ravel()

def retarget(world, previous):
    # Convert MP camera coordinates to a right-handed z-up frame.
    # Proper rotation (det=+1): MP right/down/away -> forward/left/up.
    # Negating all three axes would reflect the skeleton and reverse knee flexion.
    p = world[:, [2,0,1]] * [-1,1,-1]
    left = unit((p[23]-p[24]) * [1,1,0])
    up = np.array([0.,0.,1.])
    forward = unit(np.cross(left, up))
    basis = np.array([forward,left,up]).T
    p = (p - (p[23]+p[24])/2) @ basis
    target = [unit((p[11]+p[12]-p[23]-p[24])/2)]
    for h,k,a,s,e,w in ((23,25,27,11,13,15),(24,26,28,12,14,16)):
        target += [unit(p[k]-p[h]),unit(p[a]-p[k]),unit(p[e]-p[s]),unit(p[w]-p[e])]
    target = np.array(target).ravel()
    def residual(q):
        t = fk(q)
        # Both soles stay close to horizontal; limb direction fits dominate.
        soles = np.concatenate([(t[f"{s}_foot_link"][:3,:3]-np.eye(3)).ravel() for s in ("left","right")])
        yaw = q[[i for i,n in enumerate(JOINTS) if "Yaw" in n]]
        return np.r_[features(q)-target, .35 * soles, .30*yaw, .12*(q-previous), .04*(q-DEFAULT)]
    return least_squares(residual, np.clip(previous, LOWER+1e-6, UPPER-1e-6), bounds=(LOWER+1e-6,UPPER-1e-6), max_nfev=12).x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT/"docs/reference/sidestep_0_7.mp4")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT/"docs/reference/sidestep_motion.json")
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--min-visibility", type=float, default=.45)
    args = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or source_fps <= 0:
        raise SystemExit("Cannot decode reference video")
    width, height = int(cap.get(3)), int(cap.get(4))
    raw_overlay = args.out.with_name("sidestep_pose_overlay_raw.mp4")
    writer = cv2.VideoWriter(str(raw_overlay), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width,height))
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.model)), running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5)
    times, worlds, images, confidence = [], [], [], []
    index = 0
    with vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok or index/source_fps >= 7:
                break
            t = index/source_fps
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), round(t*1000))
            score = 0.
            if result.pose_world_landmarks:
                pose = result.pose_landmarks[0]
                score = min(min(pose[k].visibility, pose[k].presence) for k in KEY)
                for a,b in EDGES:
                    cv2.line(frame,(int(pose[a].x*width),int(pose[a].y*height)),(int(pose[b].x*width),int(pose[b].y*height)),(0,255,130),2)
                if score >= args.min_visibility:
                    times.append(t)
                    worlds.append([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]])
                    images.append([[p.x,p.y,p.z] for p in pose])
                    confidence.append(score)
            cv2.putText(frame, f"{t:.2f}s confidence={score:.2f}", (20,height-24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            writer.write(frame)
            index += 1
    writer.release()
    cap.release()
    import subprocess
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(raw_overlay),"-an","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(args.out.with_name("sidestep_pose_overlay.mp4"))],check=True)
    raw_overlay.unlink()
    if len(times)<20 or len(times)/index < .65:
        raise SystemExit(f"Insufficient reliable poses: {len(times)}/{index}; inspect overlay")
    max_gap = max(np.diff(times), default=0)
    if times[0]>.2 or times[-1]<6.7 or max_gap>.25:
        raise SystemExit(f"Untracked reference interval: start={times[0]}, end={times[-1]}, gap={max_gap}; no long-gap interpolation allowed")
    raw = {"time":times,"world":worlds,"image":images,"confidence":confidence,"source_fps":source_fps,"clip_end":7.0}
    args.out.with_name("sidestep_landmarks.json").write_text(json.dumps(raw))
    target_t = np.arange(0,7,1/args.fps)
    w = np.array(worlds)
    interpolated = np.stack([np.interp(target_t,times,w[:,j,k]) for j in range(33) for k in range(3)],axis=1).reshape(-1,33,3)
    q, previous = [], DEFAULT.copy()
    for i, pose in enumerate(interpolated):
        previous = retarget(pose, previous)
        q.append(previous)
        if i%20==0:
            print(f"IK {i}/{len(target_t)}",flush=True)
    q = savgol_filter(np.array(q),9,2,axis=0)
    image = np.array(images)
    ankle_width = abs(image[:,27,0]-image[:,28,0])
    neutral_frames = ankle_width <= np.quantile(ankle_width,.25)
    # Single-camera world depth has a static bias. Anchor narrow-stance frames
    # to this robot's default stance while preserving the extracted excursions.
    neutral_q = np.median(q[np.interp(target_t,times,neutral_frames.astype(float))>.5],axis=0)
    q -= neutral_q-DEFAULT
    # Limit target speed at the native URDF joint speed (10 rad/s).
    q = np.clip(q,LOWER,UPPER)
    for i in range(1,len(q)):
        q[i] = np.clip(q[i],q[i-1]-10/args.fps,q[i-1]+10/args.fps)
    hips = (image[:,23,:2]+image[:,24,:2])/2
    # Image width/height differ: use pixel displacement, scaled by 0.30m shoulder width.
    shoulder_px = np.median(np.linalg.norm((image[:,11,:2]-image[:,12,:2])*[width,height],axis=1))
    scale = .30/max(shoulder_px,1)
    lateral = -(hips[:,0]-hips[0,0])*width*scale
    root = np.array([[0.,np.interp(t,times,lateral),grounded_height(p)] for t,p in zip(target_t,q)])
    root[:,1] = savgol_filter(root[:,1],9,2)
    result = {"schema":1,"source":"mediapipe","source_url":"https://youtu.be/KKhF516rSyo","clip_start":0,"clip_end":7,
        "video_sha256":hashlib.sha256(args.video.read_bytes()).hexdigest(),"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"robot_urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "quality":{"accepted_frames":len(times),"total_frames":index,"coverage":len(times)/index,"max_gap_s":float(max_gap),"min_visibility":args.min_visibility},
        "neutral_calibration":{"method":"median lowest-quartile image ankle separation -> native default stance","offset_rad":(neutral_q-DEFAULT).tolist()},
        "limitations":["Monocular MediaPipe is an approximate motion prior, not calibrated motion capture.","Root lateral scale assumes 0.30m shoulder width and a fixed front camera.","Root height is grounded native-axis FK on source CAD sole vertices.","IK limb directions and horizontal soles are prioritized; torso roll saturates at native 0.20rad."],
        "joint_names":list(JOINTS),"time":target_t.tolist(),"q":q.tolist(),"root":root.tolist()}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["quality"],indent=2))

if __name__=="__main__":
    main()
