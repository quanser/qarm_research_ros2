# Changelog
All notable changes to this repository are recorded here. Versions follow the release tags.

## [0.1.0] - 2026-11-17

Initial release.

### Added

- `qarm_research_hardware` package: the QArm Research hardware interface and the RGB-D camera node.
- `qarm_research_rl_deploy` package: a pick and place demo driven by a trained reinforcement-learning
  reach policy.
- `qarm_research_rl_interfaces` package: the `ReachPose` action and the `ObservationStamped` message
  used by the RL demo.
- `docs/rl_pick_and_place.md`: a guide to running the RL demo from a cold machine.
- Example trained policy 2026-11-10_17-35-10.zip, attached to the release.
