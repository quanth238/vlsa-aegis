# Ellipsoid-feature contact-ranker E05 diagnostic

This test directly answers two local questions: whether a learned ranker can
select a contact-free candidate, and whether its negative contact-risk
gradient produces a contact-free action. It calculates no mesh distance.

Inputs are the robot joint state, OSC goal, obstacle pose, seven accepted
L5--L7 ellipsoid margins against the unchanged released-AEGIS obstacle MVEE,
the nominal Cartesian action, and the candidate Cartesian action. Targets are
three binary flags recording any nonpositive L5, L6, or L7 obstacle contact
over every internal MuJoCo step of a full cloned OSC transition. Obstacle
motion and ellipsoid margins remain diagnostics; only raw protected contact
defines the classifier target and fresh pass/fail result.

Complete states are split before training: 184/185/186/188/189/191 train,
192 validation, and 187/190 test. The validation state sets a threshold just
below its least-risk unsafe candidate. Best-of-N must have zero false-safe
test candidates and freshly verify the least-change predicted-safe action at
both test states. Separately, bounded actions at 0.1/0.25/0.5/1.0 along the
negative predicted-risk gradient must freshly verify at both states. Ranking
GO with gradient NO-GO supports only candidate selection. Neither outcome
establishes closed-loop task success or unseen-episode generalization.
