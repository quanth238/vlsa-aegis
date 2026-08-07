# AEGIS Cartesian-to-joint-velocity bridge pilot preregistration

## Question

Can the task-competent released π0.5+AEGIS Cartesian action ledger remain
competent when its translational commands are executed through a direct
seven-joint-velocity bridge, and can the same bridge with exactly three
L5/L6/L7 ellipsoid CBF rows prevent the primary physical-link collision?

This is a single-case controller pilot inside active gate E02. It does not
modify or reinterpret Table 1 and does not authorize KKT/VI training.

## Frozen pairing

Both new arms restore the immutable primary state and replay the same 237
released AEGIS `env.step` inputs. They use the same 20 Hz control frequency,
20 settling actions, frozen obstacle MVEE, and exact archived gripper command.
The only paired difference is whether the three link constraints are applied
to the physical joint velocity that is sent to the controller.

The bridge clips archived XYZ to `[-1, 1]`, interprets the released
translational protocol as zero angular velocity, and maps an action unit to
`0.2 m/s` before damped resolved-rate inversion. The scale is frozen before
execution: the already archived primary trajectory gives per-axis
least-squares realized scales `0.183683`, `0.240975`, and `0.224089 m/s`,
whose mean is `0.216249 m/s`; `0.2 m/s` is the preregistered isotropic rounded
value. No scale, gain, clearance, or threshold will be changed after outcome
inspection.

Both arms apply the same `1 rad/s` joint-velocity bounds, one-step joint-range
margin, and exact direct `JOINT_VELOCITY` controller. The active arm minimizes
deviation from the bridge velocity under one simultaneous QP with exactly
three rigid-link rows. `D_opt=0.01 m` remains distinct from raw-contact and
obstacle-displacement `D_sim` evidence.

## Decision rule

The bridge-only arm must both satisfy the native goal and reproduce the
primary protected-link contact and paper CAR failure. If it does not, the
active arm cannot support a safety-efficacy conclusion. Conditional on that
competence gate, the active arm passes only if it satisfies the native goal,
has no L5/L6/L7 contact, and never exceeds `0.001 m` obstacle displacement.
All infeasible QPs and complete videos are retained.
