# Open with the ME team

- `WAIST_SHOULD_HEAD` mass is wrong: `ROBOT_CHEST_ASM` at 4.1254508e+10 kg/m3 (waist_should_head.m_p.4 L50) gives 5.36e7 kg. We use a 5.36 kg box placeholder.
- `ANKLE_ROLL_L` / `ANKLE_ROLL_R`: no .m_p ever delivered. We use a 0.2 kg box placeholder.
- `FEET_PITCH_L` / `FEET_PITCH_R`: no .m_p ever delivered. We use a 0.5 kg box placeholder.
- AK45-10 not measured: `arm_l_link_4` / `arm_r_link_4` still on the 0921 data.
- AK45-36 not measured: neck, and no neck link exists in the URDF yet.
- `THIGH_HIP_YAW_L/R` mass differs left vs right: 1.839 vs 1.935 kg for the same part.
- `arm_l_link_4` vs `arm_r_link_4` CoM is not mirrored: y = -0.0195 vs +0.0296.
- `ARM_L_LINK_6_JOINT` y = 0.0001 while the right side is 0 - intended or noise?
- The 20260929 sheet reverted the arm values to the 0904 ones; we use the 0921 values.
- `ARM_R_LINK_1` x is +0.0302 in the sheet, should be -0.0302 (confirmed typo, fixed on our side).
- Rows 23-26 of the 0921 sheet are labelled ARM_L but belong to ARM_R.
- Waist order in the sheet is pitch then roll; the CAD is yaw then roll then pitch.
- `THIGH_HIP_YAW` is a roll joint and `THIGH_KNEE_PITCH` is a yaw joint - the row names state the wrong axis.
- `arm_l_link_4.STL` and `arm_r_link_2/3/4.stl` are exported in mm, everything else in m.
- `LOWER_LEG_*_LINK_1_2` lists an "other joint" at (0, +/-0.046, -0.09) that nothing attaches to.
- No .m_p for the head; no mesh for the head either.

# Open on our side

- `generate_pdf.py` is slow (~800k triangles projected and sorted per view).
- Joint limits other than the arms are ours, not the ME's - they have never confirmed them.
- `check_inertia.py` only proves the CoM is inside the mesh, not that the STL origin is on the real rotation axis.
