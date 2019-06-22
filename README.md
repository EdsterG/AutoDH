## AutoDH

[![Build Status](https://travis-ci.org/EdsterG/AutoDH.svg?branch=master)](https://travis-ci.org/EdsterG/AutoDH)
[![Coverage Status](https://coveralls.io/repos/github/EdsterG/AutoDH/badge.svg?branch=master)](https://coveralls.io/github/EdsterG/AutoDH?branch=master)

A library for automatic extraction of Denavit-Hartenberg parameters from a robot model.

### Install
To install AutoDH run:
```
$ pip install autodh
```

### OpenRAVE Script
To generate DH parameters from OpenRAVE models use `scripts/openrave.py`.

Here's an example for Puma from link `Base` to link `Puma6`:
```
$ python scripts/openrave.py robots/pumaarm.zae Base Puma6
```
Another example for PR2 from link `base_link` to link `r_gripper_palm_link`:
```
$ python scripts/openrave.py robots/pr2-beta-static.zae base_link r_gripper_palm_link
```

### Run Tests
Install test suite dependencies before running unit tests:
```
$ pip install autodh[test]
$ pytest autodh
```

#### References:

- Chittawadigi et. al. “Automatic Extraction of DH Parameters of Serial Manipulators using Line Geometry.” _The 2nd International Conference on Multibody System Dynamics_, 2012.
- Lipkin H. “A Note on Denavit-Hartenberg Notation in Robotics.“ _International Design Engineering Technical Conferences and Computers and Information in Engineering Conference, Volume 7: 29th Mechanisms and Robotics Conference, Parts A and B_, 2005.