"""Inverse Kinematics solver:
Jacobian Inverse (Newton-Raphson) method 
with damping least squares(optimized by Gemini)
"""

import numpy as np
import mujoco
from typing import Optional, Tuple


class IKSolver:
    
    def __init__(
        self,
        damping: float = 0.01, 
        max_iter: int = 100,
        tolerance: float = 1e-3,
        step_size: float = 0.5
    ):
        #Initialize IK solver.
                
        self.damping = damping  #for K matrix
        self.max_iter = max_iter  #max iterations
        self.tolerance = tolerance  #when to stop
        self.step_size = step_size  
    
    def compute_ik(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_pos: np.ndarray,   #target position [x, y, z] (meters)   
        target_quat: Optional[np.ndarray] = None,  #target quaternion [w, x, y, z] (not used in this project)
        site_name: str = "ee_site",  #name of the ee (see xml file)
        joint_names: Optional[list] = None  #list of joint names to control (if None, uses first 7 joints)
    ) -> Tuple[np.ndarray, bool]:
        """       
        Returns:
            Tuple of (joint_positions, converged)
                - joint_positions: Computed joint positions
                - converged: Whether IK converged within tolerance
        """
        # Get ID of ee
        try:
            site_id = model.site(site_name).id
        except KeyError:
            print(f"Warning: Site '{site_name}' not found, using last site")
            site_id = model.nsite - 1
        
        """ test """
        # if joint_names is None:
        #     # Default to first 7 joints (arm joints, excluding gripper)
        #     joint_ids = list(range(min(7, model.nv)))
        # else:
        #     joint_ids = [model.joint(name).id for name in joint_names]

        #joints to control
        joint_ids = list(range(min(7, model.nv)))
        n_joints = len(joint_ids)
                
        # Current joint positions
        qpos = data.qpos.copy()
        
        # Iterative IK solving
        for iteration in range(self.max_iter):
            # current end-effector pose(use FK)
            mujoco.mj_forward(model, data)            
            current_pos = data.site(site_id).xpos.copy()
            
            # error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            # convergence? -> stop here
            if error_norm < self.tolerance:
                return qpos[joint_ids], True
            
            # Jacobian
            jacp = np.zeros((3, model.nv)) #pos
            jacr = np.zeros((3, model.nv)) #rot
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            
            # only need position jacobian 
            J = jacp[:, joint_ids]
            
            # If orientation is specified, include rotation error
            if target_quat is not None:
                current_quat = self._mat_to_quat(data.site(site_id).xmat)
                rot_error = self._quat_error(target_quat, current_quat)
                
                # Combine position and rotation errors
                error = np.concatenate([pos_error, rot_error])
                J_full = np.vstack([jacp[:, joint_ids], jacr[:, joint_ids]])
                J = J_full
            else:
                error = pos_error
            
            # q_dot
            delta_q = self.JacInv(J, error)
            
            # Update joint positions
            qpos[joint_ids] += self.step_size * delta_q
            
            # Clamp to joint limits
            for i, joint_id in enumerate(joint_ids):
                if joint_id < model.njnt:
                    jnt_range = model.jnt_range[joint_id]
                    if jnt_range[0] < jnt_range[1]:  # Check if limits are valid
                        qpos[joint_id] = np.clip(qpos[joint_id], jnt_range[0], jnt_range[1])
            
            # Update data with new joint positions
            data.qpos[:] = qpos
        
        # Did not converge
        return qpos[joint_ids], False
    
    def JacInv(self, J: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Jacobian Inverse
        Args:
            J: Jacobian matrix 
            error: Error vector 
            
        Returns:
            Joint displacements
        """
        m, n = J.shape
        
        # Damped Least Squares: delta_q = J^T (J J^T + λ²I)^-1 error(optimize by gemini)
        JJT = J @ J.T
        I = np.eye(m)
        damping_matrix = JJT + (self.damping ** 2) * I
        
        try:
            delta_q = J.T @ np.linalg.solve(damping_matrix, error)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if solve fails
            delta_q = np.linalg.pinv(J) @ error
        
        return delta_q  #discretized joint displacements
    
   
   
    """this project doesn't invloved quaternion , 
    it was used for solving problems of scenario 1, failed eventually, so it was not used in this project"""

    def _mat_to_quat(self, mat: np.ndarray) -> np.ndarray:
        """Rotation matrix to quaternion [w, x, y, z].       
        Args:
            mat: 3x3 rotation matrix (flattened to length 9 in MuJoCo)          
        Returns:
            Quaternion [w, x, y, z]
        """
        # Reshape if needed
        if mat.shape == (9,):
            mat = mat.reshape(3, 3)
        
        # Convert using MuJoCo function
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return quat
    
    def _quat_error(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Calculate orientation error between two quaternions.       
        Args:
            q_target: Target quaternion 
            q_current: Current            
        Returns:
            Error vector (3,) representing rotation needed
        """
        # Compute error quaternion: q_error = q_target * q_current^-1
        q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Quaternion multiplication
        q_error = self._quat_mult(q_target, q_current_inv)
        
        # Extract rotation vector (imaginary part scaled by 2)
        error = 2.0 * q_error[1:4]
        
        return error
    
    def _quat_mult(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

