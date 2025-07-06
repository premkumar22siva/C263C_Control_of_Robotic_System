# This file shows our attempt at implementing impedance control for scara robot to assemble Lego block. 
# The code complies and executes succesfully, however behavious of system under its control is not stable. 

import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime
import scipy
import numpy as np
from dxl import (
    DynamixelMode, 
    DynamixelModel, 
    DynamixelMotorGroup, 
    DynamixelMotorFactory, 
    DynamixelIO
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray
#from scipy.interpolate import PPoly

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel

#Impedance control for 3 DoF Scara
class Scara_ImpedanceControl:
    def __init__(
        self, 
        motor_group: DynamixelMotorGroup,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        M_D: NDArray[np.double],
        q_initial_deg: Sequence[float] = None,
        
        max_duration_s: float = 50.0,
    ):

        self.q_initial_rad = np.deg2rad(q_initial_deg)
        #self.x_desired = np.array(x_desired, dtype=np.double)  # Desired (x, y, z)
        self.K_P = np.asarray(K_P, dtype=np.double)  # (3, 3)
        self.K_D = np.asarray(K_D, dtype=np.double)  # (3, 3)
        self.M_D = np.asarray(M_D, dtype=np.double)  # (3, 3)
    
        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        self.joint_position_history = deque()
        self.joint_velocity_history = deque()
        self.time_stamps = deque()
        self.motor_group = motor_group

        self.x_desired = np.zeros((4,))     # [x, y, z, yaw]
        self.xd_desired = np.zeros((4,))
        self.xdd_desired = np.zeros((4,))
        # Load spline data for all 4 DOFs (1-based index)
        self.trajectory_splines = [self.load_spline_set(self, i+1) for i in range(4)]

        # Robot parameters (edit as needed)
        self.l1 = 0.0675
        self.l2 = 0.07
        self.m1, self.m2, self.m3 = 0.193537, 0.0156075, 
        self.lc1, self.lc2 = 0.0533903, 0.0281188

        # Structural parameters of the robot
        self.support_z = 0.1588 #As given by Deb
        self.gear_radius = 0.015 #m radius of the pinion at motor 3
        self.station_frameZ = 0.057

        # Environment compliance 
        self.K_e = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 500, 0]]) #Initial set value 500 N/m

        # DC motor model (reuse from your skeleton)
        self.pwm_limits = np.array([info.pwm_limit for info in self.motor_group.motor_info.values()])
        self.motor_model = DCMotorModel(self.control_period_s, pwm_limits=self.pwm_limits)

        # Clean exit handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    # reading ref trajectory
    def load_spline_set(self, dof: int):
        #"""Load position, velocity, and acceleration splines for a given DOF (1-based index)."""
        def load_one(kind):
            breaks = np.loadtxt(f'spline_dof_{kind}{dof}_breaks.csv')
            coefs = np.loadtxt(f'spline_dof_{kind}{dof}_coefs.csv').T  # shape (4, N)
            return PPoly(coefs, breaks)

        pp_pos = load_one('pos')
        pp_vel = load_one('vel')
        pp_acc = load_one('acc')
        return pp_pos[:3], pp_vel[:3], pp_acc[:3]
    
    # Main control loop which will run the hardware
    def start_control_loop(self):
        self.go_to_home_configuration()

        start_time = time.time()
        while self.should_continue:
            # --------------------------------------------------------------------------
            # Step 1 - Get feedback
            # --------------------------------------------------------------------------
            # Read position feedback (and covert resulting dict into to NumPy array)
            q_rad_motor = np.asarray(list(self.motor_group.angle_rad.values())) 
            q_rad = q_rad_motor - np.array([np.pi, np.pi, np.pi])

            # TODO: Read Data from Multiple Dynamixels – Joint Velocities (Question 2)
            #    Use the example above for retreiving joint position feedback (`q_rad`)
            #    and the `DynamixelMotorGroup.velocity_rad_per_s` property to extract 
            #    the joint velocity feedback in units of rad/s.
            qdot_rad_per_s_motor = (
                np.asarray(list(self.motor_group.velocity_rad_per_s.values()))
            )
            qdot_rad_per_s = np.array([qdot_rad_per_s_motor[0],
                                       qdot_rad_per_s_motor[1],
                                       self.map_rot2lin(self, qdot_rad_per_s_motor[2])])

            self.joint_position_history.append(q_rad)  # Save for plotting
            self.joint_velocity_history.append(qdot_rad_per_s)
            self.time_stamps.append(time.time() - start_time)  # Save for plotting
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 2 - Check termination criterion
            # --------------------------------------------------------------------------
            # Stop after 2 seconds
            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return
            # --------------------------------------------------------------------------
            # Computing necessary matrices
            G_q = self.calc_gravity_compensation_torque(self, q_rad, cos_q2) #Gravity compensation
            C_q_qdot = self.calc_christoffle(self, q_rad, qdot_rad_per_s) #Coriolis compensation
            B_q = self.calc_robotInertia(self, q_rad) # Instantaneous robot inertia
            Ja = self.calc_analyticJacob(self, q_rad) #4*3
            inv_Ja = self.calc_analyticJacob(self, q_rad) #3*4
            dot_Ja = self.calc_analyticJacobDot(self, q_rad, qdot_rad_per_s) #4*3
            
            #Computing End Effector pose
            e_pose = self.fkine_scara(self, q_rad)
            e_velocity = Ja @ qdot_rad_per_s.reshape(-1,1)

            # --------------------------------------------------------------------------
            # Step 3 - Compute error term in operational
            # --------------------------------------------------------------------------
            # TODO: Compute Error Term (Question 3)
            # Use the `self.q_desired` variable and the `q_actual` variable to compute
            # the joint position error for the current time step.
            
            # --------------------------------------------------------------------------
            #Computing operation space coordinates
            x_error = self.desired_pose - e_pose
            xdot_error = self.desired_velocty - e_velocity
            xddot_desired = self.desired_acc
            cos_q2 = np.cos(q_rad[1])
            # --------------------------------------------------------------------------
            # Step 4 - Calculate control action
            # --------------------------------------------------------------------------
            # Computing force from the environment
            if (e_pose[2] > self.station_frameZ):
                force_endEff = 0
            else:
                force_endEff = self.K_e@(e_pose.reshape(-1,1) - self.station_frameZ) #Did we define 

            u_theoretical = self.M_D@xddot_desired + self.K_P@x_error + self.K_D@xdot_error 
            u_theoretical = u_theoretical -  force_endEff - self.M_D @ dot_Ja @ qdot_rad_per_s #Check the sign of force_endEff
            u_theoretical = B_q @ inv_Ja @ np.linalg.inv(self.M_D) @ u_theoretical
            u_theoretical = u_theoretical + G_q + C_q_qdot 

            u = np.array([u_theoretical[0],
                          u_theoretical[1],
                          self.tau_lin2rot(self, u_theoretical[2])]) #Passed to motors

            # --------------------------------------------------------------------------
            # Step 5 - Command control action
            # --------------------------------------------------------------------------
            # This code converts the torque control action into a PWM command using a
            # model of the dynamixel motors
            pwm_command = self.motor_model.calc_pwm_command(u)

            # TODO:  Sending Joint PWM Commands (Question 4)
            # Replace "..." with the calculated `pwm_command` variable
            self.motor_group.pwm = {
                dxl_id: pwm_value
                for dxl_id, pwm_value in zip(
                    self.motor_group.dynamixel_ids, pwm_command, strict=True
                )
            }
            # --------------------------------------------------------------------------


            # Print current position in degrees
            print("q [deg]:", np.degrees(q_rad))

            # This code helps this while loop run at a fixed frequency
            self.loop_manager.sleep()

        self.stop()

    #function for Going to home position
    def go_to_home_configuration(self):
        """Puts the motors in 'home' position"""
        self.should_continue = True
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        self.q_initial_rad = self.q_initial_rad + np.array([np.pi, np.pi, np.pi])

        # Move to home position (self.q_initial)
        home_positions_rad = {
            dynamixel_id: self.q_initial_rad[i]
            for i, dynamixel_id in enumerate(self.motor_group.dynamixel_ids)
        }
        
        self.motor_group.angle_rad = home_positions_rad
        time.sleep(0.5)
        abs_tol = math.radians(1.0)
        
        should_continue_loop = True
        while should_continue_loop:
            should_continue_loop = False
            q_rad = self.motor_group.angle_rad
            for dxl_id in home_positions_rad:
                if abs(home_positions_rad[dxl_id] - q_rad[dxl_id]) > abs_tol:
                    should_continue_loop = True
                    break
            

        # Set PWM Mode (i.e. voltage control)
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()
    
    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    ############ Function for Computing the control ##################
    # Defining the desired trajectory    
    def trajectory(self, start_time):
        t = time.time() - start_time
        t = min(t, self.trajectory_splines[0][0].x[-1])  # clip time to last breakpoint

        # Evaluate all 4 DOFs
        x_pos = np.array([s[0](t) for s in self.trajectory_splines])     # position
        x_vel = np.array([s[1](t) for s in self.trajectory_splines])     # velocity
        x_acc = np.array([s[2](t) for s in self.trajectory_splines])     # acceleration

        # Store for controller use
        self.x_desired = x_pos
        self.xd_desired = x_vel
        self.xdd_desired = x_acc

        return self.x_desired, self.xd_desired, self.xdd_desired

    def tau_lin2rot(self, 
                prism_joint_Force: float,
        ) -> NDArray[np.double]:
        
        return prism_joint_Force*self.gear_radius

    # g(q): Computing gravity compensation for scara (zero compensation)
    def calc_gravity_compensation_torque(
        self, joint_positions_rad: NDArray[np.double],
        ) -> NDArray[np.double]:

        return np.array([0, 0, 0])

    # C(q, q_dot): Matrix of Christoffle coefficients
    def calc_christoffle(
            self, joint_positions_rad: NDArray[np.double],
            joint_velocity_rad: NDArray[np.double] 
        ) -> NDArray[np.double]:

        return np.zeros((3,3))

    # M(q): Robot inertia matrix
    def calc_inertia_mat(self, joint_position_rad: NDArray[np.double], cos_q1: float
                         ) -> NDArray[np.double]:
        #Compute the 3x3 inertia matrix M(q) for the 2R1P SCARA robot.
        #The third joint is prismatic with negligible mass modeled as self.mass_link3.
        
        q1, q2, q3 = joint_position_rad
        cos_q2 = np.cos(q2)

        # Link lengths and COM positions
        l1, l2 = self.l1, self.l2
        lc1, lc2 = self.lc1, self.lc2
        m1, m2 = self.m1, self.m2

        # Rotational inertias (assuming mass at COM)
        I1 = m1 * lc1**2
        I2 = m2 * lc2**2

        # Upper 2x2 block for 2R part
        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_q2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * cos_q2)
        M22 = I2 + m2 * lc2**2

        # Build 3x3 inertia matrix
        M = np.zeros((3, 3), dtype=np.double)
        M[0, 0] = M11
        M[0, 1] = M[1, 0] = M12
        M[1, 1] = M22
        M[2, 2] = self.mass_link3  # Prismatic link mass

        return M

    # Forward Kinematics
    def map_rot2lin(self, kinVar):
        #kinVar is any kinematic variable of the 3rd motor
        return self.gear_radius*kinVar

    def fkine_scara(self, q_rad) -> NDArray[np.double]:
        q1, q2, q3 = q_rad #Measurement from motor
        x = self.l1*np.cos(q1) + self.l2*np.cos(q1+q2)
        y = self.l2*np.sin(q1) + self.l2*np.sin(q1+q2)
        z = self.support_z - self.map_rot2lin(q3) 
        yaw = q1 + q2
        return np.array([x, y, z, yaw])


    # J_A(q): Analytical jacobian
    def calc_analyticJacob(
        self, joint_positions_rad: NDArray[np.double] 
        ) -> NDArray[np.double]:

        q1, q2, q3 = joint_positions_rad
        J_A = np.zeros((4,3))
        J_A[0,0] = -self.l1*np.sin(q1) -self.l1*np.sin(q1 + q2)
        J_A[0,1] = -self.l2*np.sin(q1 + q2)
        J_A[0,2] = 0
        J_A[1,0] = self.l1*np.cos(q1) +self.l1*np.cos(q1 + q2)
        J_A[1,1] = self.l2*np.cos(q1 + q2)
        J_A[1,2] = 0
        J_A[2,0] = 0
        J_A[2,1] = 0
        J_A[2,2] = 1
        J_A[3,0] = 1
        J_A[3,1] = 1
        J_A[3,2] = 0
        return J_A

    # inv_J_A(q): Analytical jacobian inverse
    def calc_invAnalyticJacob(
        self, joint_positions_rad: NDArray[np.double] 
        ) -> NDArray[np.double]:

        J_A = self.calc_analyticJacob(self, joint_positions_rad) 
        U, s, Vt = np.linalg.svd(J_A, full_matrices = False)
        S_pinv = np.diag(1.0/s)
        J_A_pinv = Vt.T@ S_pinv @U.T
        return  np.linalg.inv(J_A_pinv)

    # J_A(q)_dot: Analytical jacobian
    def calc_analyticJacobDot(
        self, joint_positions_rad: NDArray[np.double],
        joint_velocities: NDArray[np.double]
        ) -> NDArray[np.double]:

        q1, q2, q3 = joint_positions_rad
        q1_dot, q2_dot, q3_dot = joint_velocities
        q12_dot = q1_dot + q2_dot
        J_A_dot = np.zeros((4,3))
        J_A_dot[0,0] = -self.l1*np.cos(q1)*q1_dot - self.l1*np.cos(q12_dot)
        J_A_dot[0,1] = -self.l2*np.cos(q1+q2)*q12_dot 
        J_A_dot[1,0] = -self.l1*np.sin(q1)*q1_dot - self.l2*np.sin(q12_dot)
        J_A_dot[1,1] = -self.l2*np.sin(q1+q2)*q12_dot 
        
        return J_A_dot



if __name__ == "__main__":
    
    # A Python list with two elements representing the initial joint configuration
    q_initial = [0, 0, 0]

    # A numpy array of shape (2, 2) representing the proportional gains of your
    # controller
    K_P = np.array([[1, 0, 0], 
                    [0, 1.2, 0], 
                    [0, 0, 0.1]])

    # A numpy array of shape (2, 2) representing the derivative gains of your controller
    K_D = np.array([[0.1, 0, 0], 
                    [0, 0.002, 0], 
                    [0, 0, 0.00002]])

    M_D = np.array([[0.1, 0, 0],
                    [0, 0.1, 0],
                    [0, 0, 0.1]])


    # ----------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------
    # Create `DynamixelIO` object to store the serial connection to U2D2
    dxl_io = DynamixelIO(
        device_name="COM4",
        baud_rate=57_600,
    )

    # Create `DynamixelMotorFactory` object to create dynamixel motor object
    motor_factory = DynamixelMotorFactory(
        dxl_io=dxl_io,
        dynamixel_model=DynamixelModel.MX28
    )

    # TODO: Replace "..." below with the correct Dynamixel IDs found from Dynamixel Wizard 
    #       (in order of closest to base frame first)
    dynamixel_ids = 5, 2, 4 #Add the ID for the 3rd motor

    motor_group = motor_factory.create(*dynamixel_ids)

    # Make controller
    controller = Scara_ImpedanceControl(
        motor_group=motor_group,
        K_P=K_P,
        K_D=K_D,
        M_D=M_D,
        q_initial_deg=q_initial,
    )

    # Run controller
    controller.start_control_loop()

    # Extract results
    time_stamps = np.asarray(controller.time_stamps)
    joint_positions = np.rad2deg(controller.joint_position_history).T
    print(np.array(joint_positions).shape)
