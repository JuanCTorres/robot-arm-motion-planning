# Robot arm motion planning

How to run?

`python3 src/RobotArmMotion.py` will generate random start and end configurations for a 3-arm robot. Since these
configurations are random and the obstacles form a pretty tight space, there is a good chance there will not be a 
solution with the conservative parameter `k=5`. Give it a few tries.  