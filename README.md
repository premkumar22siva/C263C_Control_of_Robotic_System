# Control-of-3-DOF-Robotic-Manipulator-for-Lego-Block-Assembly

This project was done for the course 263C - Control of Robotic Systems

We have designed and developed a 3-DOF robotic manipulator for the task of Lego block assembly. We have developed an inverse dynamics controller that ensures the smooth operation of picking up the Lego blocks from an initial position, bringing them to a final position, and subsequently press-fitting the block on the Lego bed, using a magnet-fitted rack-and-pinion mechanism. We show the performance of our designed controller in smoothly following a reference trajectory. We also compare the performance of the proposed controller for various sets of gains for the same task. The present study only focuses on joint-space inverse dynamics control, which considers the interaction forces from the end-effector as a disturbance. The same application with force control is left for the future.

To run the code, run the setup file first, configure your dynamixel motors and assembly, and then execute the run file by opening either the impedance control source code or inverse dynamics control source code. The trajectory via points to be used are within the trajectory folder.
