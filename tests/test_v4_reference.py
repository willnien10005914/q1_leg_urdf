"""Physical-source invariants for the new native-axis reference library."""
import numpy as np

from leg_humanoid_lab.robot import JOINTS, LOWER, UPPER, FOOT_VERTICES, fk
from leg_humanoid_lab.v4_reference import V4Reference, WALK_BASES


def test_v4_native_limits_velocity_and_periodic_seams():
    reference = V4Reference()
    for name, plan in reference.data["plans"].items():
        q = np.asarray(plan["q"])
        assert q.shape == (61, 23)
        assert np.all(q >= LOWER-1e-7) and np.all(q <= UPPER+1e-7)
        assert np.max(np.abs(np.diff(q,axis=0))) / (plan["duration_s"]/60) < 10.
        assert np.allclose(q[0], q[-1], atol=1e-7), name


def test_v4_both_sidestep_feet_lift_and_opposite_hand_reaches_front():
    reference = V4Reference()
    for side, active, direction in (("left","right",1),("right","left",-1)):
        plan = reference.data["plans"]["sidestep_"+side]
        peaks = np.zeros(2)
        for q, root in zip(plan["q"],plan["root_offset_body"]):
            transforms = fk(q)
            for index, foot in enumerate(("left","right")):
                transform = transforms[foot+"_foot_link"]
                clearance = np.min(FOOT_VERTICES[foot]@transform[2,:3]+transform[2,3])+reference.root_height+root[2]
                peaks[index] = max(peaks[index],clearance)
        assert min(peaks) > .035
        midpoint = fk(reference.sample_sidestep(side,.5)["q"])
        hand = midpoint[active+"_gripper_link"][:3,3]
        assert hand[0] > .20 and hand[1]*direction > .06
        assert np.allclose(plan["q"][0],reference.default)
        assert np.allclose(plan["q"][-1],reference.default)
        assert np.allclose(plan["root_offset_body"][-1],[.085,direction*.085,0])


def test_v4_convex_direction_mix_is_bounded_and_zero_command_is_stand():
    reference = V4Reference()
    for command in ([0,0,0],[.22,0,0],[-.22,.18,.6],[4,-4,-4]):
        weights,effective = reference.walk_weights(command)
        assert np.all(weights >= -1e-12) and np.isclose(weights.sum(),1)
        samples = reference.sample_walk(command,.3)
        assert np.all(samples["q"] >= LOWER-1e-7)
        assert np.all(samples["q"] <= UPPER+1e-7)
        basis = np.asarray([reference.data["basis_command"][name] for name in WALK_BASES])
        assert np.allclose(effective,weights@basis)
    assert np.allclose(reference.sample_walk([0,0,0],.3)["q"],reference.default)


def test_slow_walk_keeps_true_cad_swing_clearance_and_native_speed_limits():
    reference = V4Reference()
    for command in ([.055,0,0],[0,-.045,0],[0,0,.15],[.0275,.0225,0]):
        peaks=np.zeros(2);positions=[]
        for phase in np.linspace(0,1,121):
            sample=reference.sample_walk(command,phase)
            q=sample['q'];positions.append(q)
            assert np.all(q>=LOWER-1e-7) and np.all(q<=UPPER+1e-7)
            transforms=fk(q)
            for index,foot in enumerate(('left','right')):
                transform=transforms[foot+'_foot_link']
                clearance=np.min(FOOT_VERTICES[foot]@transform[2,:3]+transform[2,3])+reference.root_height+sample['root_offset_body'][2]
                peaks[index]=max(peaks[index],clearance)
        assert min(peaks)>.035, (command,peaks)
        assert np.max(np.abs(np.diff(positions,axis=0)))/(reference.walk_cycle_s/120)<10.
