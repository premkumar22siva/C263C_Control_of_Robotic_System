import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from scipy.interpolate import PPoly

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

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel
def load_joint_splines(num_joints=3, base_names=('pos', 'vel', 'acc')):
    spline_dict = {name: [] for name in base_names}
    
    for name in base_names:
        for j in range(num_joints):
            coefs = np.loadtxt(f"spline_joint_{name}{j+1}_coefs.csv", delimiter=",")
            breaks = np.loadtxt(f"spline_joint_{name}{j+1}_breaks.csv", delimiter=",")
            spline = PPoly(c=coefs.T, x=breaks)
            spline_dict[name].append(spline)
    
    return spline_dict
      
# PD with Gravity Compensation Controller from Minilab
class JSID_Controller:
    def __init__(
        self,
        motor_group: DynamixelMotorGroup,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        max_duration_s: float = 50.0,
        pos_splines=None,
        vel_splines=None,
        acc_splines=None,
        T_total=None
    ):
        # ------------------------------------------------------------------------------
        # Controller Related Variables
        # ------------------------------------------------------------------------------
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)

        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)

        self.control_freq_Hz = 30.0 
        self.max_duration_s = float(max_duration_s)
        
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True
        self.pos_splines = pos_splines
        self.vel_splines = vel_splines
        self.acc_splines = acc_splines

        self.joint_position_history = deque()
        self.time_stamps = deque()
        self.torque_history = deque()
        self.velocity_history = deque()
        self.acceleration_history = deque()
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # Manipulator Parameters
        # ------------------------------------------------------------------------------
        self.m1, self.m2, self.m3 = 0.193537, 0.0156075, 0.005
        self.lc1, self.lc2, self.lc3 = 0.0675, 0.08733, 0.05 # 0.0533903, 0.0281188, 0.025
        self.l1, self.l2, self.l3 = 0.0675, 0.08733, 0.05    # l3 is wrist length

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # Motor Communication Related Variables
        # ------------------------------------------------------------------------------
        self.motor_group: DynamixelMotorGroup = motor_group

        # ------------------------------------------------------------------------------
        # DC Motor Modeling
        # ------------------------------------------------------------------------------
        self.pwm_limits = []
        for info in self.motor_group.motor_info.values():
            self.pwm_limits.append(info.pwm_limit)
        self.pwm_limits = np.asarray(self.pwm_limits)

        # This model is based on the DC motor model learned in class, it allows us to
        # convert the torque control action u into something we can actually send to the
        # MX28-AR dynamixel motors (pwm voltage commands).
        self.motor_model = DCMotorModel(
            self.control_period_s, pwm_limits=self.pwm_limits
        )
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Clean Up / Exit Handler Code
        # ------------------------------------------------------------------------------
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        # ------------------------------------------------------------------------------
        
    def start_control_loop(self):
        self.go_to_home_configuration()

        start_time = time.time()
        while self.should_continue:
            # --------------------------------------------------------------------------
            # Step 1 - Get feedback
            # --------------------------------------------------------------------------
            # Read position feedback (and covert resulting dict into to NumPy array)
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))
            qdot_rad_per_s = (
                np.asarray(list(self.motor_group.velocity_rad_per_s.values()))
            )

            self.joint_position_history.append(q_rad)  # Save for plotting
            self.time_stamps.append(time.time() - start_time)  # Save for plotting
            self.velocity_history.append(qdot_rad_per_s.copy())

            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 2 - Check termination criterion
            # --------------------------------------------------------------------------
            # Stop after 2 seconds
            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 3 - Compute error term
            # --------------------------------------------------------------------------
            # TODO: Compute Error Term (Question 3)
            # Use the `self.q_desired` variable and the `q_actual` variable to compute
            # the joint position error for the current time step.
            # q_error = self.q_desired_rad - q_rad
            t_now = time.time() - start_time
            q_des_now = np.array([spline(t_now) for spline in self.pos_splines])[[1, 2, 0]]
            q_des_now[1] /= -0.015  # Convert prismatic to equivalent revolute motion
            qd_des_now = np.array([spline(t_now) for spline in self.vel_splines])[[1, 2, 0]]
            qd_des_now[1] /= -0.015  # Convert prismatic to equivalent revolute motion
            qdd_des_now = np.array([spline(t_now) for spline in self.acc_splines])[[1, 2, 0]]
            qdd_des_now[1] /= -0.015  # Convert prismatic to equivalent revolute motion

 

            q_error = q_des_now +np.array([np.pi,np.pi,np.pi])- q_rad 
            q_error = q_error[[2, 0, 1]]
            print('q_error =', q_error)
            qdot_error = qd_des_now - qdot_rad_per_s
            qdot_error = qdot_error[[2, 0, 1]]
            print('qdot_error =', qdot_error)
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 4 - Calculate control action
            # --------------------------------------------------------------------------
            gravity_comp_torques = self.calc_gravity_compensation_torque(q_rad)
            print('G_q = ', gravity_comp_torques)
            M_q = self.calc_inertia_mat(q_rad)
            print('M_q = ', M_q)
            # TODO: Calculate Control Law (Question 3)
            # Use the `self.K_P`, `q_error`, `self.K_D`, `q_dot_actual`, and
            # `gravity_comp_torques` variables to compute the control action for joint
            # space PD control with gravity compensation.
            #
            # Note: This is a torque control action!
            # u = self.K_P@q_error - self.K_D@qdot_rad_per_s + gravity_comp_torques 

            # JSID Control Law
            u = M_q @ (qdd_des_now + self.K_P @ q_error + self.K_D @ qdot_error) + gravity_comp_torques
            print('Control torques = ', u)
            # u = np.array([u[1], u[2], u[0]])  # Reorder torque command for motor mapping
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 5 - Command control action
            # --------------------------------------------------------------------------
            # This code converts the torque control action into a PWM command using a
            # model of the dynamixel motors
            u = u[[1,2,0]]
            pwm_command = self.motor_model.calc_pwm_command(u)
            self.torque_history.append(u.copy())
            if len(self.velocity_history) > 1:
                acc_est = (self.velocity_history[-1] - self.velocity_history[-2]) / self.control_period_s
            else:
                acc_est = np.zeros_like(qdot_rad_per_s)
            self.acceleration_history.append(acc_est)

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
            print("q [deg]:", np.degrees(q_rad-np.array([np.pi,np.pi,np.pi])))

            # This code helps this while loop run at a fixed frequency
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def calc_gravity_compensation_torque(
        self, joint_positions_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2, q3 = joint_positions_rad
      
        from math import cos
        g = 9.81

        m1, m2,m3 = self.m1, self.m2, self.m3
        l1 = self.l1
        lc1, lc2 = self.lc1, self.lc2

        #tau1, tau2 will be zero because structure supposrts the gravity load of first two links
        #tau3 is the weight of the third link (prismatic joint)
        tau1= 0*(m1 * g * lc1 * cos(q1) + m2 * g * (l1 * cos(q1) + lc2 * cos(q1 + q2)))
        tau2= 0*( m2 * g * lc2 * cos(q1 + q2))
        tau3= m3*g
        return np.array([tau1, tau2, tau3])
         
    def calc_inertia_mat(self, joint_position_rad: NDArray[np.double]
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
        M11 = I1 + I2 + m2 * (lc1**2+ 2 * l1 * lc2 * cos_q2)
        M12 = I2 + m2 * (  lc1 * lc2 * cos_q2)
        M22 = I2 + m2 * lc2**2

        # Build 3x3 inertia matrix
        M = np.zeros((3, 3), dtype=np.double)
        M[0, 0] = M11
        M[0, 1] = M[1, 0] = M12
        M[1, 1] = M22
        M[2, 2] = self.m3  # Prismatic link mass

        return M

    def go_to_home_configuration(self):
        """Puts the motors in 'home' position"""
        self.should_continue = True
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        # Move to home position (self.q_initial)
        home_positions_rad = {
            dynamixel_id: self.q_initial_rad[i]
            for i, dynamixel_id in enumerate(self.motor_group.dynamixel_ids)
        }
        
        self.motor_group.angle_rad = home_positions_rad
        time.sleep(3.5)
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


if __name__ == "__main__":
    # A Python list with two elements representing the initial joint configuration # motor order is 2-4-5 in actual
    q_initial = [180,180,180] # motor order is 2-4-5 in actual

    # A Python list with two elements representing the desired joint configuration
    q_desired = [90,180,180]

    # #

    ##################################################
    # A numpy array of shape (3, 3) representing the proportional gains of your controller
    K_P = np.array([[75, 0, 0], [0, 120, 0], [0, 0, 100]])
    print('K_P = ', K_P)
    # A numpy array of shape (2, 2) representing the derivative gains of your controller
    K_D = np.array([[3, 0, 0], [0, 0.2, 0],  [0, 0, 0.2]])
    print('K_D = ', K_D)




    # ----------------------------------------------------------------------------------
    # Create `DynamixelIO` object to store the serial connection to U2D2
    dxl_io = DynamixelIO(
        device_name="COM4",
        baud_rate=57_600,
    )

    motor_factory = DynamixelMotorFactory(
        dxl_io=dxl_io,
        dynamixel_model=DynamixelModel.MX28
    )

    dynamixel_ids = 2,4,5
    motor_group = motor_factory.create(*dynamixel_ids)

    splines = load_joint_splines()
    pos_splines, vel_splines, acc_splines = splines['pos'], splines['vel'], splines['acc']

    T_total = pos_splines[0].x[-1]

    controller = JSID_Controller(
        motor_group=motor_group,
        K_P=K_P,
        K_D=K_D,
        q_initial_deg=q_initial,
        q_desired_deg=q_desired,
        pos_splines=pos_splines,
        vel_splines=vel_splines,
        acc_splines=acc_splines,
        T_total=T_total
    )
    # ----------------------------------------------------------------------------------

    # Run controller
    controller.start_control_loop()

    # ----------------------------------------------------------------------------------
    #Collecting trajectory data
    
    time_stamps = np.asarray(controller.time_stamps)
    ref_q = np.array([[spline(t) for spline in pos_splines] for t in time_stamps])[:, [1, 2, 0]]
    ref_q[:, 1] /= 0.015
    ref_qd = np.array([[spline(t) for spline in vel_splines] for t in time_stamps])[:, [1, 2, 0]]
    ref_qd[:, 1] /= 0.015
    ref_qdd = np.array([[spline(t) for spline in acc_splines] for t in time_stamps])[:, [1, 2, 0]]
    ref_qdd[:, 1] /= 0.015

    actual_velocities = np.array(controller.velocity_history)
    actual_velocities[:, 1] *= 0.015  # Convert revolute to prismatic for Joint 3
    actual_velocities = np.array(controller.velocity_history).T
    actual_accelerations = np.array(controller.acceleration_history)
    if actual_accelerations.shape[0] != len(time_stamps):
        actual_accelerations = actual_accelerations[:len(time_stamps)]

    actual_accelerations = np.array(actual_accelerations).T
    joint_positions = np.rad2deg(controller.joint_position_history).T
    torque_log = np.array(controller.torque_history)
    torque_log[:, 1] /= 0.015
    date_str = datetime.now().strftime("%d-%m_%H-%M-%S")

#Plotting
fig_file_name = f"joint_positions_vs_time_{date_str}.pdf"

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
labels = ["Motor 2 (Joint 2 -Revolute)", "Motor 4 (Joint 3 - Prismatic)", "Motor 5 (Joint 1 -Revolute)"]

# Adjust the joint positions for plotting
for i in range(3):
    q_deg = joint_positions[i] - 180.0
    ref_deg = np.rad2deg(ref_q[:, i]) if i != 1 else ref_q[:, i]*0.015
    if i == 1:
        q_deg = -np.radians(q_deg) * 0.015
    axs[i].plot(time_stamps, q_deg, 'k-', label='Actual')
    axs[i].plot(time_stamps, ref_deg, 'r--', label='Reference')
    axs[i].set_title(labels[i])
    axs[i].set_xlabel('Time [s]')
    axs[i].set_ylabel('Displacement [m]' if i == 1   else 'Angle [deg]')
    axs[i].legend()
    axs[i].grid(True)
plt.suptitle('Motor Trajectories (Adjusted for Offset and Units)')
plt.tight_layout()
plt.savefig(f"joint_positions_vs_time_{date_str}.pdf")

# Plot velocities
fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 10))
for i in range(3):
    axs_vel[i].plot(time_stamps, actual_velocities[i], 'k-', label='Actual Velocity')
    ref_vel = ref_qd[:, i] if i != 1 else ref_qd[:, i] * 0.015
    axs_vel[i].plot(time_stamps, ref_vel, 'r--', label='Ref Velocity')
    axs_vel[i].set_title(f'Joint {i+1} Velocity')
    axs_vel[i].set_xlabel('Time [s]')
    axs_vel[i].set_ylabel('Velocity [m/s]' if i ==1 else 'Velocity [rad/s]')
    axs_vel[i].legend()
    axs_vel[i].grid(True)
fig_vel.suptitle('Joint Velocities')
plt.tight_layout()
fig_vel.savefig(f'velocities_vs_time_{date_str}.pdf')

# Plot accelerations
fig_acc, axs_acc = plt.subplots(3, 1, figsize=(10, 10))
for i in range(3):
    axs_acc[i].plot(time_stamps[:-1], actual_accelerations[i], 'k-', label='Actual Acceleration')
    axs_acc[i].plot(time_stamps[:-1], ref_qdd[:, i], 'r--', label='Ref Acceleration')
    axs_acc[i].set_title(f'Joint {i+1} Acceleration')
    axs_acc[i].set_xlabel('Time [s]')
    axs_acc[i].set_ylabel('Accel [m/s²]' if i ==1 else 'Accel [rad/s²]')
    axs_acc[i].legend()
    axs_acc[i].grid(True)
fig_acc.suptitle('Joint Accelerations')
plt.tight_layout()
fig_acc.savefig(f'accelerations_vs_time_{date_str}.pdf')


# Plot joint torques or forces
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10))
torque_labels = ['Joint 2 Torque', 'Joint 3 Force (Converted)', 'Joint 1 Torque']
for i in range(3):
    axs2[i].plot(time_stamps[:len(torque_log)], torque_log[:, i], label=torque_labels[i], color='orange')
    axs2[i].set_title(torque_labels[i])
    axs2[i].set_xlabel('Time [s]')
    axs2[i].set_ylabel('Force [N]' if i ==1 else 'Torque [Nm]')
    axs2[i].legend()
    axs2[i].grid(True)
fig2.suptitle('Joint Torques and Forces')
plt.tight_layout()
fig2.savefig(f"torques_vs_time_{date_str}.pdf")


# Save actual data
np.savetxt(f"actual_joint_log_{date_str}.txt", np.column_stack([
    time_stamps,
    joint_positions.T,
    actual_velocities.T
    # actual_accelerations.T
]), header="time pos1 pos2 pos3 vel1 vel2 vel3")

# Save reference data
np.savetxt(f"reference_joint_log_{date_str}.txt", np.column_stack([
    time_stamps,
    ref_q,
    ref_qd
    # ref_qdd
]), header="time ref_pos1 ref_pos2 ref_pos3 ref_vel1 ref_vel2 ref_vel3")