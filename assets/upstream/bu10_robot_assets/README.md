
# quanta_robot

    meshes/    STL
    urdf/      xacro + generated URDF
    ME_data/   xlsx: link_dim / joint_axes / inertia
    tools/     scripts below

Every tool writes into output/<tool_name>/.
First time on a new PC:  pip install -r requirements.txt

# visual
visit https://viewer.robotsfan.com/ and upload this whole folder there (not just the URDF folder or you will see no visual meshes)

## tools

    python3 tools/build_urdf.py legged
Builds urdf/quanta_robot_legged.urdf from the xacro, and checks every link puts its mesh at 0,0,0.
Argument: `wheeled`, `legged`, or nothing = both.

    python3 tools/generate_pdf.py legged
Draws output/pdf/quanta_robot_legged.pdf from the xlsx + STL (does not read the URDF).
Argument: `wheeled`, `legged`, or nothing = both.
TODO : Super slow

    python3 tools/check_symmetry.py --plot
Compares every left STL to its right twin. `--plot` writes output/symmetry/lr_diff_<link>.png.

    python3 tools/check_inertia.py --plot
Checks mass, CoM and inertia in the xlsx against the meshes. Exits 1 if something is wrong.
`--plot` writes output/inertia/<link>.png showing the CoM against the mesh.

