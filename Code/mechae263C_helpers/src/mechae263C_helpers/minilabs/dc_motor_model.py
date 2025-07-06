from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MotorParams:
    K: float
    J: float
    L: float
    R: float
    b: float


class DCMotorModel:
    def __init__(
        self,
        sample_period_s: float,
        pwm_limits: NDArray[np.int64],
        motor_params: MotorParams = MotorParams(
            K=1.777643817803365,
            J=0.2870505805441941,
            L=0.095073355742376,
            R=0.20960884501675667,
            b=1.7447233413813388,
        ),
    ):
        self.motor_params = motor_params
        self.sample_period_s = sample_period_s
        self.pwm_limits = np.abs(np.asarray(pwm_limits, dtype=np.int64))
        num_dofs = len(pwm_limits)

        K, J, L, R, b = (
            self.motor_params.K,
            self.motor_params.J,
            self.motor_params.L,
            self.motor_params.R,
            self.motor_params.b,
        )
        Ts = sample_period_s

        JL = J * L
        JR_Lb_Ts = (J * R + L * b) * Ts
        KK_Rb_TsTs = (K * K + R * b) * Ts * Ts

        denom = (Ts * b + J) * K * Ts
        self.b = np.array([J, JR_Lb_Ts + 2 * JL, JL + KK_Rb_TsTs + JR_Lb_Ts]) / denom
        self.a = np.array([J]) / (Ts * b + J)

        self.torque_history = deque(
            [
                np.zeros((num_dofs,), dtype=np.double),
                np.zeros((num_dofs,), dtype=np.double),
                np.zeros((num_dofs,), dtype=np.double),
            ],
            maxlen=3,
        )
        self.voltage_history = deque([np.zeros((num_dofs,), dtype=np.double)], maxlen=1)

    def calc_pwm_command(self, torque_command: NDArray[np.double]) -> NDArray[np.int64]:
        self.torque_history.append(torque_command)
        voltage_command = self.b.dot(self.torque_history) - self.a.dot(
            self.voltage_history
        )
        self.voltage_history.append(voltage_command)

        return np.minimum(
            self.pwm_limits,
            np.maximum(-self.pwm_limits, self.voltage_to_pwm_command(voltage_command)),
        )

    def voltage_to_pwm_command(self, voltage: NDArray[np.double]) -> NDArray[np.int64]:
        return np.round(voltage * self.pwm_limits / 12.3).astype(np.int64)


class DCMotorModelStateEstimator:
    def __init__(
        self,
        motor_params: MotorParams = MotorParams(
            K=1.777643817803365,
            J=0.2870505805441941,
            L=0.095073355742376,
            R=0.20960884501675667,
            b=1.7447233413813388,
        ),
    ):
        self.motor_params = motor_params

        K, J, L, R, b = (
            self.motor_params.K,
            self.motor_params.J,
            self.motor_params.L,
            self.motor_params.R,
            self.motor_params.b,
        )

        self.state_matrix = np.asarray(
            [
                [-b / J, K / J],
                [-K / L, -R / L],
            ],
            dtype=np.double,
        )
        self.input_matrix = np.asarray([[0], [1 / L]], dtype=np.double)
        self.output_matrix = np.asarray([[1, 0]], dtype=np.double)

        self.state = np.zeros((2, 1), dtype=np.double)
        self.covariance = np.eye(2, dtype=np.double)

        self.process_noise_covariance = np.diag([1, 1]).astype(np.double)
        self.measurement_covariance = np.array([[1.0]], dtype=np.double)

    def update(self, new_voltage_command: float, joint_speed: float):
        # Predict (Time Update)
        state_estimate = (
            self.state_matrix @ self.state + self.input_matrix * new_voltage_command
        )
        covariance_estimate = (
            self.state_matrix @ self.covariance @ self.state_matrix.T
            + self.process_noise_covariance
        )

        # Correct (Measurement Update)
        kalman_gain = (
            covariance_estimate
            @ self.output_matrix.T
            @ np.linalg.inv(
                self.output_matrix @ covariance_estimate @ self.output_matrix.T
                + self.measurement_covariance
            )
        )

        joint_speed = np.asarray([[joint_speed]])
        self.state = state_estimate + kalman_gain @ (
            joint_speed - self.output_matrix @ state_estimate
        )
        _1 = np.eye(2) - kalman_gain @ self.output_matrix
        self.covariance = (
            _1 @ covariance_estimate @ _1.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )

    @property
    def torque(self) -> float:
        return float(self.motor_params.K * self.state[1])
