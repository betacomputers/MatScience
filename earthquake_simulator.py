#!/usr/bin/env python3
"""
Earthquake Simulator - Physics-Based Implementation

This simulator implements earthquake dynamics consistent with the physical principles:
1. Ground acceleration (a) is imposed on the structure
2. Seismic force: F = m × a (Newton's Second Law)
3. Base shear: V = C × W (where W = weight = m × g)
4. Natural frequency: f = (1/2π) × √(k/m)
5. Kinetic energy: KE = ½ × m × v²
6. Stress: σ = F/A = (m × a)/A

Reducing mass directly:
- Reduces seismic forces (F = ma)
- Reduces base shear (V = C × W)
- Increases natural frequency (f ∝ 1/√m)
- Reduces kinetic energy (KE ∝ m)
- Reduces stress in structural elements (σ = F/A = ma/A)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass

# Import existing simulation infrastructure
from mazars_model import (
    MaterialProperties,
    SimulationParameters,
    load_stl_and_create_mesh,
    compute_equivalent_strain,
    mazars_compressive_damage,
    mazars_tensile_damage
)

try:
    from sfepy.discrete.fem import FEDomain
    SFEPY_AVAILABLE = True
except ImportError:
    SFEPY_AVAILABLE = False
    print("Warning: SfePy not available. Some features may be limited.")


@dataclass
class GroundMotion:
    """Ground motion record (seismic accelerogram)"""
    time: np.ndarray
    acceleration: np.ndarray
    dt: float
    duration: float
    pga: float  # Peak Ground Acceleration
    name: str = "ground_motion"
    
    @classmethod
    def from_array(cls, time: np.ndarray, acceleration: np.ndarray, name: str = "ground_motion"):
        """Create ground motion from time and acceleration arrays"""
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        duration = time[-1] - time[0]
        pga = np.max(np.abs(acceleration))
        return cls(time, acceleration, dt, duration, pga, name)
    
    @classmethod
    def synthetic(cls, duration: float = 20.0, dt: float = 0.01, 
                  pga: float = 4.9, frequency_range: Tuple[float, float] = (1.0, 15.0),
                  name: str = "synthetic"):
        """Generate synthetic ground motion"""
        time = np.arange(0, duration, dt)
        n = len(time)
        
        # Generate broadband motion using filtered white noise
        white_noise = np.random.randn(n)
        
        # Apply frequency filter
        freqs = np.fft.fftfreq(n, dt)
        fft_signal = np.fft.fft(white_noise)
        
        # Bandpass filter
        mask = (np.abs(freqs) >= frequency_range[0]) & (np.abs(freqs) <= frequency_range[1])
        fft_signal[~mask] = 0
        
        # Inverse FFT
        acceleration = np.real(np.fft.ifft(fft_signal))
        
        # Normalize to target PGA
        acceleration = acceleration / np.max(np.abs(acceleration)) * pga
        
        # Apply envelope (build-up and decay)
        envelope = np.exp(-(time - duration/2)**2 / (2 * (duration/4)**2))
        envelope = envelope / np.max(envelope)
        acceleration = acceleration * envelope
        
        return cls.from_array(time, acceleration, name)


@dataclass
class EarthquakeSimulationParameters:
    """Parameters for earthquake simulation"""
    ground_motion: GroundMotion
    damping_ratio: float = 0.05  # 5% damping
    damping_frequencies: Tuple[float, float] = (1.0, 10.0)  # For Rayleigh damping
    output_frequency: int = 50  # Save results every N steps (increased for speed)
    damage_update_frequency: int = 10  # Update damage every N steps (increased for speed)
    element_size: float = 0.005  # Mesh element size in meters


class EarthquakeSimulator:
    """
    Physics-based earthquake simulator implementing:
    - Ground acceleration imposition
    - Mass-proportional seismic forces (F = ma)
    - Base shear calculation (V = C × W)
    - Natural frequency (f = (1/2π) × √(k/m))
    - Kinetic energy (KE = ½ × m × v²)
    - Stress calculation (σ = F/A = ma/A)
    """
    
    def __init__(self, domain, material: MaterialProperties, 
                 sim_params: EarthquakeSimulationParameters):
        """Initialize earthquake simulator"""
        self.domain = domain
        self.material = material
        self.sim_params = sim_params
        self.ground_motion = sim_params.ground_motion
        
        # Initialize state variables
        self.displacement_history = []  # Relative displacement (internal deformation)
        self.absolute_displacement_history = []  # Absolute displacement (total movement)
        self.velocity_history = []
        self.absolute_velocity_history = []
        self.acceleration_history = []
        self.absolute_acceleration_history = []
        self.ground_displacement_history = []  # Ground displacement (integrated from acceleration)
        self.ground_velocity_history = []  # Ground velocity (integrated from acceleration)
        self.damage_history = []
        self.time_history = []
        self.stress_history = []
        self.base_shear_history = []
        self.kinetic_energy_history = []
        self.natural_frequency_history = []
        
        # Setup boundary conditions first (needed for natural frequency calculation)
        self._setup_boundary_conditions()
        
        # Compute mass and stiffness matrices
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """Initialize mass, damping, and stiffness matrices"""
        n_nodes = self.domain.mesh.n_nod
        n_dof = n_nodes * 3
        
        print(f"\nMatrix Assembly:")
        print(f"  Nodes: {n_nodes}")
        print(f"  DOF: {n_dof}")
        
        # Mass matrix (lumped)
        self.M = self._assemble_mass_matrix()
        
        # Initial stiffness matrix
        self.K0 = self._assemble_stiffness_matrix(damage=None)
        
        # Damping matrix (Rayleigh damping: C = αM + βK)
        omega1 = 2 * np.pi * self.sim_params.damping_frequencies[0]
        omega2 = 2 * np.pi * self.sim_params.damping_frequencies[1]
        alpha = 2 * self.sim_params.damping_ratio * (omega1 * omega2) / (omega1 + omega2)
        beta = 2 * self.sim_params.damping_ratio / (omega1 + omega2)
        self.C = alpha * self.M + beta * self.K0
        
        # Calculate total mass and weight
        self.total_mass = np.trace(self.M)  # Total mass in kg
        g = 9.81  # Gravitational acceleration (m/s²)
        self.total_weight = self.total_mass * g  # Weight in N
        
        # Calculate natural frequency: f = (1/2π) × √(k/m)
        # For multi-DOF system, use first mode
        if len(self.free_dof) > 0:
            K_free = self.K0[np.ix_(self.free_dof, self.free_dof)]
            M_free = self.M[np.ix_(self.free_dof, self.free_dof)]
            # Solve generalized eigenvalue problem: Kφ = λMφ
            # Use proper generalized eigenvalue solver
            try:
                # For symmetric matrices, use eigh (more stable than eig)
                # scipy.linalg.eigh is better, but numpy.linalg.eigh works for symmetric
                # Since M is diagonal (lumped), we can use: M^(-1/2) K M^(-1/2) φ = λ φ
                # Or solve: (M^-1 K) φ = λ φ (but this can be ill-conditioned)
                
                # Better approach: use Cholesky decomposition for stability
                # M = L L^T, then solve: L^-1 K L^-T y = λ y
                try:
                    from scipy.linalg import eigh
                    eigenvals, _ = eigh(K_free, M_free)
                    eigenvals = np.real(eigenvals[eigenvals > 0])
                except ImportError:
                    # Fallback: use numpy (less stable but available)
                    # M is diagonal, so M^-1 is easy
                    M_inv = np.diag(1.0 / np.diag(M_free))
                    eigenvals, _ = np.linalg.eig(M_inv @ K_free)
                    eigenvals = np.real(eigenvals[eigenvals > 0])
                
                if len(eigenvals) > 0:
                    omega_n = np.sqrt(np.min(eigenvals))  # First natural frequency (rad/s)
                    self.natural_frequency = omega_n / (2 * np.pi)  # Convert to Hz
                else:
                    # Fallback: approximate using average stiffness and mass
                    k_avg = np.mean(np.diag(K_free))
                    m_avg = np.mean(np.diag(M_free))
                    self.natural_frequency = (1 / (2 * np.pi)) * np.sqrt(k_avg / m_avg) if m_avg > 0 else 0.0
            except Exception as e:
                # Fallback: approximate using average stiffness and mass
                k_avg = np.mean(np.diag(K_free))
                m_avg = np.mean(np.diag(M_free))
                self.natural_frequency = (1 / (2 * np.pi)) * np.sqrt(k_avg / m_avg) if m_avg > 0 else 0.0
                print(f"  Warning: Eigenvalue solve failed, using approximation: {e}")
        else:
            self.natural_frequency = 0.0
        
        print(f"  Total mass: {self.total_mass:.6f} kg")
        print(f"  Total weight: {self.total_weight:.6f} N ({self.total_weight/1000:.2f} kN)")
        print(f"  Natural frequency: {self.natural_frequency:.3f} Hz")
        print(f"  Damping ratio: {self.sim_params.damping_ratio*100:.1f}%")
    
    def _assemble_mass_matrix(self) -> np.ndarray:
        """Assemble lumped mass matrix with proper volume calculation"""
        n_nodes = self.domain.mesh.n_nod
        n_dof = n_nodes * 3
        
        mesh = self.domain.mesh
        coors = mesh.coors
        
        # Get element connectivity
        conn = self._get_connectivity()
        if conn is None or len(conn) == 0:
            # Fallback: calculate from bounding box
            size = np.ptp(coors, axis=0)
            volume = np.prod(size)
            total_mass = self.material.rho * volume
            mass_per_node = total_mass / n_nodes
            M = np.eye(n_dof) * mass_per_node
            print(f"  Using fallback mass calculation: {total_mass:.6f} kg")
            return M
        
        # Calculate total volume properly by summing element volumes
        total_volume = 0.0
        element_volumes = []
        
        for el_conn in conn:
            if len(el_conn) < 4:  # Need at least 4 nodes for 3D element
                continue
            el_coors = coors[el_conn]
            
            # For hexahedral elements (8 nodes), calculate volume using Gauss quadrature
            # For tetrahedral (4 nodes), use formula: V = |det(J)|/6
            if len(el_conn) == 4:
                # Tetrahedron: V = |det([v1-v0, v2-v0, v3-v0])| / 6
                v0, v1, v2, v3 = el_coors[0], el_coors[1], el_coors[2], el_coors[3]
                J = np.array([v1-v0, v2-v0, v3-v0]).T
                el_volume = abs(np.linalg.det(J)) / 6.0
            elif len(el_conn) == 8:
                # Hexahedron: use Gauss quadrature to integrate volume
                gauss_points, gauss_weights = self._get_gauss_quadrature_3d(2)
                el_volume = 0.0
                for gp, weight in zip(gauss_points, gauss_weights):
                    xi, eta, zeta = gp
                    N, dN_dxi = self._hex8_shape_functions(xi, eta, zeta)
                    J = dN_dxi.T @ el_coors
                    det_J = np.linalg.det(J)
                    if det_J > 0:
                        el_volume += det_J * weight
            else:
                # Fallback: bounding box
                el_size = np.ptp(el_coors, axis=0)
                el_volume = np.prod(el_size)
            
            element_volumes.append((el_conn, el_volume))
            total_volume += el_volume
        
        # Calculate total mass
        total_mass = self.material.rho * total_volume
        
        # Distribute mass to nodes (each node gets mass from all elements it belongs to)
        node_mass = np.zeros(n_nodes)
        for el_conn, el_volume in element_volumes:
            el_mass = self.material.rho * el_volume
            # Distribute equally to all nodes in element
            mass_per_node = el_mass / len(el_conn)
            for node_idx in el_conn:
                node_mass[node_idx] += mass_per_node
        
        # Create lumped mass matrix (diagonal)
        M = np.zeros((n_dof, n_dof))
        for i in range(n_nodes):
            M[i*3, i*3] = node_mass[i]
            M[i*3+1, i*3+1] = node_mass[i]
            M[i*3+2, i*3+2] = node_mass[i]
        
        # Verify total mass
        calculated_total = np.sum(node_mass)
        print(f"  Total volume: {total_volume:.9f} m³")
        print(f"  Total mass: {calculated_total:.6f} kg (expected: {total_mass:.6f} kg)")
        
        return M
    
    def _assemble_stiffness_matrix(self, damage: Optional[np.ndarray] = None) -> np.ndarray:
        """Assemble stiffness matrix using proper FE assembly"""
        n_nodes = self.domain.mesh.n_nod
        n_dof = n_nodes * 3
        
        # Get effective Young's modulus (degraded by damage)
        if damage is None:
            E_eff = np.full(n_nodes, self.material.E)
        else:
            if len(damage) == n_nodes:
                E_eff = self.material.E * (1.0 - damage)
            else:
                E_eff = np.full(n_nodes, self.material.E * (1.0 - np.mean(damage)))
        
        nu = self.material.nu
        
        # Initialize global stiffness matrix
        K = np.zeros((n_dof, n_dof))
        
        mesh = self.domain.mesh
        coors = mesh.coors
        
        # Get element connectivity
        conn = self._get_connectivity()
        if conn is None or len(conn) == 0:
            # Fallback: approximate stiffness
            size = np.ptp(coors, axis=0)
            L = np.max(size)  # Characteristic length
            # Approximate: k ≈ E * A / L where A ≈ L²
            k_approx = np.mean(E_eff) * L
            K = np.eye(n_dof) * k_approx
            return K
        
        # Gauss quadrature
        gauss_points, gauss_weights = self._get_gauss_quadrature_3d(2)
        
        # Process each element
        for el_conn in conn:
            if len(el_conn) < 4:
                continue
            
            el_coors = coors[el_conn]
            n_nodes_el = len(el_conn)
            n_dof_el = n_nodes_el * 3
            
            # Average E for element
            el_E = np.mean(E_eff[el_conn])
            lambda_lame = el_E * nu / ((1 + nu) * (1 - 2 * nu))
            mu_lame = el_E / (2 * (1 + nu))
            D = self._compute_material_matrix(lambda_lame, mu_lame)
            
            K_e = np.zeros((n_dof_el, n_dof_el))
            
            # Integrate over element
            for gp, weight in zip(gauss_points, gauss_weights):
                xi, eta, zeta = gp
                
                # Shape functions
                if n_nodes_el == 4:
                    N, dN_dxi = self._tet4_shape_functions(xi, eta, zeta)
                else:
                    N, dN_dxi = self._hex8_shape_functions(xi, eta, zeta)
                
                # Jacobian
                J = dN_dxi.T @ el_coors
                det_J = np.linalg.det(J)
                
                if det_J <= 0:
                    continue
                
                J_inv = np.linalg.inv(J)
                dN_dx = dN_dxi @ J_inv.T
                
                # Build B matrix (strain-displacement)
                B = np.zeros((6, n_dof_el))
                for inode in range(n_nodes_el):
                    idx = inode * 3
                    dN_dx_i = dN_dx[inode, 0]
                    dN_dy_i = dN_dx[inode, 1]
                    dN_dz_i = dN_dx[inode, 2]
                    
                    B[0, idx] = dN_dx_i      # ε_xx
                    B[1, idx + 1] = dN_dy_i  # ε_yy
                    B[2, idx + 2] = dN_dz_i  # ε_zz
                    B[3, idx] = dN_dy_i      # γ_xy
                    B[3, idx + 1] = dN_dx_i
                    B[4, idx] = dN_dz_i      # γ_xz
                    B[4, idx + 2] = dN_dx_i
                    B[5, idx + 1] = dN_dz_i  # γ_yz
                    B[5, idx + 2] = dN_dy_i
                
                # Element stiffness: K_e = ∫ B^T D B dV
                if n_nodes_el == 4:
                    dV = det_J / 6.0  # Tetrahedron volume
                else:
                    dV = det_J * weight  # Hexahedron
                
                K_e += B.T @ D @ B * dV
            
            # Assemble into global matrix
            for i, inode in enumerate(el_conn):
                for j, jnode in enumerate(el_conn):
                    i_dof = np.arange(inode * 3, inode * 3 + 3)
                    j_dof = np.arange(jnode * 3, jnode * 3 + 3)
                    i_local = np.arange(i * 3, i * 3 + 3)
                    j_local = np.arange(j * 3, j * 3 + 3)
                    K[np.ix_(i_dof, j_dof)] += K_e[np.ix_(i_local, j_local)]
        
        return K
    
    def _get_connectivity(self):
        """Get element connectivity from mesh"""
        mesh = self.domain.mesh
        if hasattr(mesh, 'conns') and len(mesh.conns) > 0:
            return mesh.conns[0]
        elif hasattr(mesh, 'get_conn'):
            try:
                return mesh.get_conn('3_8')
            except:
                try:
                    return mesh.get_conn('3_4')
                except:
                    return None
        return None
    
    def _get_gauss_quadrature_3d(self, n_points: int = 2):
        """Get Gauss quadrature points and weights for 3D hexahedral elements"""
        if n_points == 2:
            xi_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w_1d = np.array([1.0, 1.0])
        else:
            xi_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w_1d = np.array([1.0, 1.0])
        
        points = []
        weights = []
        for i, xi in enumerate(xi_1d):
            for j, eta in enumerate(xi_1d):
                for k, zeta in enumerate(xi_1d):
                    points.append([xi, eta, zeta])
                    weights.append(w_1d[i] * w_1d[j] * w_1d[k])
        
        return np.array(points), np.array(weights)
    
    def _tet4_shape_functions(self, xi: float, eta: float, zeta: float):
        """Shape functions for 4-node tetrahedral element"""
        N = np.array([
            1 - xi - eta - zeta,
            xi,
            eta,
            zeta
        ])
        
        dN_dxi = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1],
        ])
        
        return N, dN_dxi
    
    def _hex8_shape_functions(self, xi: float, eta: float, zeta: float):
        """Shape functions for 8-node hexahedral element"""
        N = np.array([
            0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 + zeta),
        ])
        
        dN_dxi = np.array([
            [-0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 - eta)],
            [ 0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 - eta)],
            [ 0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 + eta)],
            [-0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 + eta)],
            [-0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 - eta)],
            [ 0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 - eta)],
            [ 0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 + eta)],
            [-0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 + eta)],
        ])
        
        return N, dN_dxi
    
    def _compute_material_matrix(self, lambda_lame: float, mu_lame: float) -> np.ndarray:
        """Compute material matrix D for isotropic linear elasticity"""
        D = np.array([
            [lambda_lame + 2*mu_lame, lambda_lame, lambda_lame, 0, 0, 0],
            [lambda_lame, lambda_lame + 2*mu_lame, lambda_lame, 0, 0, 0],
            [lambda_lame, lambda_lame, lambda_lame + 2*mu_lame, 0, 0, 0],
            [0, 0, 0, mu_lame, 0, 0],
            [0, 0, 0, 0, mu_lame, 0],
            [0, 0, 0, 0, 0, mu_lame],
        ])
        return D
    
    def _setup_boundary_conditions(self):
        """Setup boundary conditions (fix base nodes)"""
        mesh = self.domain.mesh
        coors = mesh.coors
        n_nodes = mesh.n_nod
        n_dof = n_nodes * 3
        
        # Find base nodes (minimum z-coordinate)
        z_min = np.min(coors[:, 2])
        z_tolerance = np.ptp(coors[:, 2]) * 0.01
        base_nodes = np.where(np.abs(coors[:, 2] - z_min) < z_tolerance)[0]
        
        # Fix all DOF for base nodes
        fixed_dof = []
        for node in base_nodes:
            fixed_dof.extend([node * 3, node * 3 + 1, node * 3 + 2])
        
        self.fixed_dof = np.array(fixed_dof, dtype=int)
        self.free_dof = np.setdiff1d(np.arange(n_dof), self.fixed_dof)
        
        print(f"\nBoundary Conditions:")
        print(f"  Fixed nodes: {len(base_nodes)} ({len(self.fixed_dof)} DOF)")
        print(f"  Free DOF: {len(self.free_dof)}")
    
    def _calculate_base_shear(self, u_ddot: np.ndarray, u_g_ddot: float) -> float:
        """
        Calculate base shear force: V = C × W
        
        Where:
        - V = base shear force
        - C = seismic coefficient (depends on ground motion and building properties)
        - W = total weight of building
        
        For this implementation:
        - C is approximated from response acceleration
        - V ≈ M × (a_structure - a_ground) at base nodes
        """
        # Base shear is the sum of forces at fixed nodes
        # Force = M × (total acceleration - ground acceleration)
        total_accel = u_ddot.copy()
        total_accel[self.fixed_dof] += u_g_ddot  # Add ground acceleration
        
        # Force at each DOF: F = m × a
        forces = self.M @ total_accel
        
        # Base shear = sum of forces at base nodes
        base_shear = np.sum(np.abs(forces[self.fixed_dof]))
        
        # Alternative: V = C × W, where C = response_accel / g
        response_accel = np.max(np.abs(total_accel))
        C = response_accel / 9.81  # Seismic coefficient
        V_alt = C * self.total_weight
        
        # Return the larger of the two (conservative)
        return max(base_shear, V_alt)
    
    def _calculate_stress(self, u_ddot_absolute: np.ndarray) -> float:
        """
        Calculate stress: σ = F/A = (m × a)/A
        
        Where:
        - σ = stress
        - F = force = m × a (using absolute acceleration)
        - A = cross-sectional area
        - m = mass
        - a = absolute acceleration
        """
        # Calculate forces from absolute acceleration: F = M × a_absolute
        # M is diagonal (lumped mass), so F[i] = M[i,i] * a[i]
        forces = self.M @ u_ddot_absolute
        
        # Get structure dimensions for cross-sectional area
        mesh = self.domain.mesh
        coors = mesh.coors
        size = np.ptp(coors, axis=0)
        
        # Cross-sectional area (for cube: A = L²)
        L = np.mean(size)  # Average dimension
        A_cross_section = L * L  # Cross-sectional area
        
        # Calculate stress from forces
        # For a cube under earthquake loading, stress is approximately:
        # σ = F_total / A, where F_total is the total inertial force
        # Total force = sum of all forces (or use max component)
        
        # Method 1: Use maximum force component (from distributed forces)
        F_max = np.max(np.abs(forces))
        
        # Method 2: Use total mass × max acceleration (more physically correct)
        # This gives: F = m × a_max, σ = F / A
        a_max = np.max(np.abs(u_ddot_absolute))
        F_from_total_mass = self.total_mass * a_max
        
        # Use the larger for conservative estimate
        F_effective = max(F_max, F_from_total_mass)
        
        # Stress = Force / Area
        stress = F_effective / A_cross_section if A_cross_section > 0 else 0.0
        
        return stress
    
    def run_simulation(self) -> Dict:
        """Run earthquake simulation"""
        print("\n" + "="*60)
        print("EARTHQUAKE SIMULATION")
        print("="*60)
        print(f"Ground motion: {self.ground_motion.name}")
        print(f"Duration: {self.ground_motion.duration:.2f} s")
        print(f"PGA: {self.ground_motion.pga:.3f} m/s² ({self.ground_motion.pga/9.81:.3f} g)")
        print(f"Time steps: {len(self.ground_motion.time)}")
        
        # Initialize state
        n_dof = self.M.shape[0]
        u = np.zeros(n_dof)  # Displacement
        u_dot = np.zeros(n_dof)  # Velocity
        u_ddot = np.zeros(n_dof)  # Acceleration
        damage = np.zeros(self.domain.mesh.n_nod)
        
        # Newmark-beta parameters (average acceleration method)
        gamma = 0.5
        beta = 0.25
        
        # Time stepping
        dt = self.ground_motion.dt
        time_array = self.ground_motion.time
        ground_acc = self.ground_motion.acceleration
        
        # Initialize ground motion integration (for absolute displacement)
        u_g = 0.0  # Ground displacement
        u_g_dot = 0.0  # Ground velocity
        
        print(f"\nStarting time integration...")
        print(f"  Total time steps: {len(time_array)}")
        print(f"  Time step: {dt:.6f} s")
        start_time = time.time()
        
        # Initialize stiffness matrix
        K = self._assemble_stiffness_matrix(damage)
        damage_prev = damage.copy()  # Track previous damage to avoid unnecessary reassembly
        
        for i, t in enumerate(time_array):
            # Ground acceleration at current time
            u_g_ddot = ground_acc[i]
            
            # Integrate ground motion to get ground displacement and velocity
            # Using simple trapezoidal integration
            if i > 0:
                # Integrate acceleration to get velocity: v = v_prev + dt * (a_prev + a_curr) / 2
                u_g_dot_prev = u_g_dot
                u_g_dot = u_g_dot_prev + dt * (ground_acc[i-1] + u_g_ddot) / 2.0
                # Integrate velocity to get displacement: d = d_prev + dt * (v_prev + v_curr) / 2
                u_g_prev = u_g
                u_g = u_g_prev + dt * (u_g_dot_prev + u_g_dot) / 2.0
            else:
                u_g_dot = 0.0
                u_g = 0.0
            
            # Update stiffness matrix with current damage (only if damage changed significantly)
            if i % self.sim_params.damage_update_frequency == 0:
                # Only reassemble if damage actually changed
                if i == 0 or np.max(np.abs(damage - damage_prev)) > 1e-6:
                    K = self._assemble_stiffness_matrix(damage)
                    damage_prev = damage.copy()
                # Otherwise reuse previous K
            
            # Newmark-Beta coefficients
            a0 = 1.0 / (beta * dt**2)
            a1 = gamma / (beta * dt)
            a2 = 1.0 / (beta * dt)
            a3 = 1.0 / (2 * beta) - 1.0
            a4 = gamma / beta - 1.0
            a5 = dt * (gamma / (2 * beta) - 1.0)
            
            # Effective stiffness
            K_eff = K + a0 * self.M + a1 * self.C
            
            # SEISMIC FORCE: F = m × a (Newton's Second Law)
            # Ground acceleration is imposed on the structure
            # Force = Mass × (ground acceleration)
            F_earthquake = -self.M @ (np.ones(n_dof) * u_g_ddot)
            
            # Effective force
            F_eff = F_earthquake + self.M @ (a0 * u + a2 * u_dot + a3 * u_ddot) + \
                    self.C @ (a1 * u + a4 * u_dot + a5 * u_ddot)
            
            # Apply boundary conditions (optimized with caching)
            if len(self.fixed_dof) > 0:
                # Cache indices for faster slicing (only update when needed)
                if i == 0 or not hasattr(self, '_free_dof_cached'):
                    self._free_dof_cached = self.free_dof
                    self._fixed_dof_cached = self.fixed_dof
                
                K_eff_bc = K_eff[np.ix_(self._free_dof_cached, self._free_dof_cached)]
                F_eff_bc = F_eff[self._free_dof_cached]
                
                # Solve for free DOF
                try:
                    u_free = np.linalg.solve(K_eff_bc, F_eff_bc)
                except np.linalg.LinAlgError:
                    # Fallback: use least squares if singular
                    u_free = np.linalg.lstsq(K_eff_bc, F_eff_bc, rcond=None)[0]
                
                # Reconstruct full displacement
                u_new = np.zeros(n_dof)
                u_new[self._free_dof_cached] = u_free
                u_new[self._fixed_dof_cached] = 0.0
            else:
                # Free-standing: solve full system
                K_eff_reg = K_eff + 1e-8 * np.trace(K_eff) * np.eye(n_dof)
                try:
                    u_new = np.linalg.solve(K_eff_reg, F_eff)
                except np.linalg.LinAlgError:
                    u_new = np.linalg.lstsq(K_eff_reg, F_eff, rcond=None)[0]
            
            # Update velocity and acceleration
            u_dot_new = a1 * (u_new - u) - a4 * u_dot - a5 * u_ddot
            u_ddot_new = a0 * (u_new - u) - a2 * u_dot - a3 * u_ddot
            
            # Compute strain and update damage (only when needed)
            if i % self.sim_params.damage_update_frequency == 0:
                strain_tensor = self._compute_strain_from_displacement(u_new)
                eps_eq = compute_equivalent_strain(strain_tensor)
                
                damage_comp = mazars_compressive_damage(
                    eps_eq, self.material.epsilon_c0, 
                    self.material.A_c, self.material.B_c
                )
                damage_tens = mazars_tensile_damage(
                    eps_eq, self.material.epsilon_t0,
                    self.material.A_t, self.material.B_t
                )
                damage_new = np.maximum(damage_comp, damage_tens)
                damage = np.maximum(damage, damage_new)
            else:
                # Only compute strain if needed for stress (when storing history)
                if i % self.sim_params.output_frequency == 0:
                    strain_tensor = self._compute_strain_from_displacement(u_new)
                else:
                    strain_tensor = None  # Skip if not needed
            
            # Update state
            u = u_new
            u_dot = u_dot_new
            u_ddot = u_dot_new
            
            # Calculate absolute displacement and acceleration FIRST
            # Absolute = Relative + Ground motion
            n_nodes = self.domain.mesh.n_nod
            u_absolute = u.copy()
            u_dot_absolute = u_dot.copy()
            u_ddot_absolute = u_ddot.copy()  # FIXED: was u_dot.copy()
            
            # Add ground motion to all DOF (ground moves uniformly)
            for node_idx in range(n_nodes):
                # Add ground displacement to all 3 DOF of each node
                u_absolute[node_idx * 3] += u_g
                u_absolute[node_idx * 3 + 1] += u_g  # Assuming horizontal ground motion
                u_absolute[node_idx * 3 + 2] += u_g  # Assuming vertical ground motion
                
                # Add ground velocity
                u_dot_absolute[node_idx * 3] += u_g_dot
                u_dot_absolute[node_idx * 3 + 1] += u_g_dot
                u_dot_absolute[node_idx * 3 + 2] += u_g_dot
                
                # Add ground acceleration
                u_ddot_absolute[node_idx * 3] += u_g_ddot
                u_dot_absolute[node_idx * 3 + 1] += u_g_ddot
                u_dot_absolute[node_idx * 3 + 2] += u_g_ddot
            
            # Calculate physics-based metrics (only when storing history to save time)
            if i % self.sim_params.output_frequency == 0:
                # Base shear: V = C × W
                base_shear = self._calculate_base_shear(u_ddot, u_g_ddot)
                
                # Kinetic energy: KE = ½ × m × v²
                # For multi-DOF: KE = ½ × u_dot^T × M × u_dot
                kinetic_energy = 0.5 * u_dot.T @ self.M @ u_dot
                
                # Stress: σ = F/A = (m × a)/A (using absolute acceleration)
                stress = self._calculate_stress(u_ddot_absolute)
            else:
                base_shear = 0.0
                kinetic_energy = 0.0
                stress = 0.0  # Skip calculation
            
            # Natural frequency (only update occasionally to save time)
            if i % 500 == 0:  # Update less frequently
                if len(self.free_dof) > 0:
                    K_free = K[np.ix_(self.free_dof, self.free_dof)]
                    M_free = self.M[np.ix_(self.free_dof, self.free_dof)]
                    try:
                        # Use proper generalized eigenvalue solver
                        try:
                            from scipy.linalg import eigh
                            eigenvals, _ = eigh(K_free, M_free)
                            eigenvals = np.real(eigenvals[eigenvals > 0])
                        except ImportError:
                            # Fallback: M is diagonal, so M^-1 is easy
                            M_inv = np.diag(1.0 / np.diag(M_free))
                            eigenvals, _ = np.linalg.eig(M_inv @ K_free)
                            eigenvals = np.real(eigenvals[eigenvals > 0])
                        
                        if len(eigenvals) > 0:
                            omega_n = np.sqrt(np.min(eigenvals))
                            natural_freq = omega_n / (2 * np.pi)
                        else:
                            natural_freq = self.natural_frequency
                    except Exception as e:
                        natural_freq = self.natural_frequency
                else:
                    natural_freq = self.natural_frequency
            else:
                natural_freq = self.natural_frequency
            
            # Calculate absolute displacement and acceleration
            # Absolute = Relative + Ground motion
            n_nodes = self.domain.mesh.n_nod
            u_absolute = u.copy()
            u_dot_absolute = u_dot.copy()
            u_ddot_absolute = u_ddot.copy()
            
            # Add ground motion to all DOF (ground moves uniformly)
            for node_idx in range(n_nodes):
                # Add ground displacement to all 3 DOF of each node
                u_absolute[node_idx * 3] += u_g
                u_absolute[node_idx * 3 + 1] += u_g  # Assuming horizontal ground motion
                u_absolute[node_idx * 3 + 2] += u_g  # Assuming vertical ground motion
                
                # Add ground velocity
                u_dot_absolute[node_idx * 3] += u_g_dot
                u_dot_absolute[node_idx * 3 + 1] += u_g_dot
                u_dot_absolute[node_idx * 3 + 2] += u_g_dot
                
                # Add ground acceleration
                u_ddot_absolute[node_idx * 3] += u_g_ddot
                u_dot_absolute[node_idx * 3 + 1] += u_g_ddot
                u_ddot_absolute[node_idx * 3 + 2] += u_g_ddot
            
            # Store history
            if i % self.sim_params.output_frequency == 0:
                self.time_history.append(t)
                self.displacement_history.append(u.copy())  # Relative (internal deformation)
                self.absolute_displacement_history.append(u_absolute.copy())  # Absolute (total movement)
                self.velocity_history.append(u_dot.copy())
                self.absolute_velocity_history.append(u_dot_absolute.copy())
                self.acceleration_history.append(u_ddot.copy())
                self.absolute_acceleration_history.append(u_ddot_absolute.copy())
                self.ground_displacement_history.append(u_g)
                self.ground_velocity_history.append(u_g_dot)
                self.damage_history.append(damage.copy())
                if stress is not None:
                    self.stress_history.append(stress)
                else:
                    self.stress_history.append(0.0)
                self.base_shear_history.append(base_shear)
                self.kinetic_energy_history.append(kinetic_energy)
                self.natural_frequency_history.append(natural_freq)
            
            # Progress update (less frequent to reduce I/O overhead)
            progress_interval = max(1, min(500, len(time_array) // 20))
            if i % progress_interval == 0 or i == len(time_array) - 1:
                progress = ((i + 1) / len(time_array)) * 100
                max_disp_rel = np.max(np.abs(u))  # Relative (internal deformation)
                max_disp_abs = np.max(np.abs(u_absolute))  # Absolute (total movement)
                max_damage = np.max(damage)
                elapsed_so_far = time.time() - start_time
                rate = (i + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (len(time_array) - i - 1) / rate if rate > 0 else 0
                print(f"  {progress:.1f}%: t={t:.2f}s, "
                      f"disp_rel={max_disp_rel*1000:.4f}mm, disp_abs={max_disp_abs*1000:.2f}mm, "
                      f"max_damage={max_damage:.3f}, base_shear={base_shear/1000:.2f}kN "
                      f"(elapsed: {elapsed_so_far:.1f}s, ETA: {eta:.1f}s)")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        
        # Compute results summary
        results = self._compute_results()
        return results
    
    def _compute_strain_from_displacement(self, u: np.ndarray) -> np.ndarray:
        """Compute strain tensor from displacement"""
        mesh = self.domain.mesh
        coors = mesh.coors
        size = np.ptp(coors, axis=0)
        
        # Simplified strain calculation
        n_nodes = mesh.n_nod
        u_reshaped = u.reshape(n_nodes, 3)
        strain_avg = np.mean(np.abs(u_reshaped), axis=0) / np.max(size)
        strain_tensor = np.diag(strain_avg)
        
        return strain_tensor
    
    def _compute_results(self) -> Dict:
        """Compute comprehensive summary results"""
        if not self.time_history:
            return {}
        
        time_array = np.array(self.time_history)
        displacements = np.array(self.displacement_history)  # Relative (internal deformation)
        absolute_displacements = np.array(self.absolute_displacement_history)  # Absolute (total movement)
        velocities = np.array(self.velocity_history)
        absolute_velocities = np.array(self.absolute_velocity_history)
        accelerations = np.array(self.acceleration_history)
        absolute_accelerations = np.array(self.absolute_acceleration_history)
        ground_displacements = np.array(self.ground_displacement_history)
        damages = np.array(self.damage_history)
        stresses = np.array(self.stress_history)
        base_shears = np.array(self.base_shear_history)
        kinetic_energies = np.array(self.kinetic_energy_history)
        
        # Maximum values - Relative (internal deformation)
        max_displacement_relative = np.max(np.abs(displacements))
        max_velocity_relative = np.max(np.abs(velocities))
        max_acceleration_relative = np.max(np.abs(accelerations))
        
        # Maximum values - Absolute (total movement)
        max_displacement_absolute = np.max(np.abs(absolute_displacements))
        max_velocity_absolute = np.max(np.abs(absolute_velocities))
        max_acceleration_absolute = np.max(np.abs(absolute_accelerations))
        
        # Ground motion values
        max_ground_displacement = np.max(np.abs(ground_displacements))
        
        # Other maximum values
        max_damage = np.max(damages)
        max_stress = np.max(np.abs(stresses))
        max_base_shear = np.max(np.abs(base_shears))
        max_kinetic_energy = np.max(kinetic_energies)
        
        # Residual values
        residual_displacement_relative = np.max(np.abs(displacements[-1]))
        residual_displacement_absolute = np.max(np.abs(absolute_displacements[-1]))
        residual_damage = np.max(damages[-1])
        residual_stress = np.abs(stresses[-1])
        
        # Peak response
        peak_acceleration_relative = max_acceleration_relative
        peak_acceleration_absolute = max_acceleration_absolute
        peak_velocity_relative = max_velocity_relative
        peak_velocity_absolute = max_velocity_absolute
        
        # Structure dimensions
        mesh = self.domain.mesh
        coors = mesh.coors
        size = np.ptp(coors, axis=0)
        max_dimension = np.max(size)
        
        results = {
            # Ground motion info
            "ground_motion_name": self.ground_motion.name,
            "pga": self.ground_motion.pga,
            "pga_g": self.ground_motion.pga / 9.81,
            "duration": self.ground_motion.duration,
            
            # Mass and weight (key metrics for earthquake resistance)
            "total_mass_kg": float(self.total_mass),
            "total_weight_N": float(self.total_weight),
            "total_weight_kN": float(self.total_weight / 1000),
            
            # Natural frequency (f = (1/2π) × √(k/m))
            "natural_frequency_Hz": float(self.natural_frequency),
            
            # Base shear (V = C × W)
            "max_base_shear_N": float(max_base_shear),
            "max_base_shear_kN": float(max_base_shear / 1000),
            "base_shear_coefficient": float(max_base_shear / self.total_weight) if self.total_weight > 0 else 0.0,
            
            # Displacement metrics - Relative (internal deformation)
            "max_displacement_relative_m": float(max_displacement_relative),
            "max_displacement_relative_mm": float(max_displacement_relative * 1000),
            "residual_displacement_relative_m": float(residual_displacement_relative),
            "residual_displacement_relative_mm": float(residual_displacement_relative * 1000),
            
            # Displacement metrics - Absolute (total movement)
            "max_displacement_absolute_m": float(max_displacement_absolute),
            "max_displacement_absolute_mm": float(max_displacement_absolute * 1000),
            "residual_displacement_absolute_m": float(residual_displacement_absolute),
            "residual_displacement_absolute_mm": float(residual_displacement_absolute * 1000),
            
            # Ground motion
            "max_ground_displacement_m": float(max_ground_displacement),
            "max_ground_displacement_mm": float(max_ground_displacement * 1000),
            
            # Velocity and acceleration - Relative
            "peak_velocity_relative_m_s": float(peak_velocity_relative),
            "peak_acceleration_relative_m_s2": float(peak_acceleration_relative),
            "peak_acceleration_relative_g": float(peak_acceleration_relative / 9.81),
            
            # Velocity and acceleration - Absolute
            "peak_velocity_absolute_m_s": float(peak_velocity_absolute),
            "peak_acceleration_absolute_m_s2": float(peak_acceleration_absolute),
            "peak_acceleration_absolute_g": float(peak_acceleration_absolute / 9.81),
            
            # Kinetic energy (KE = ½ × m × v²)
            "max_kinetic_energy_J": float(max_kinetic_energy),
            
            # Stress (σ = F/A = ma/A)
            "max_stress_Pa": float(max_stress),
            "max_stress_MPa": float(max_stress / 1e6),
            "residual_stress_Pa": float(residual_stress),
            "residual_stress_MPa": float(residual_stress / 1e6),
            
            # Damage metrics
            "max_damage": float(max_damage),
            "residual_damage": float(residual_damage),
            "mean_damage": float(np.mean(damages)),
            
            # Structure info
            "max_dimension_m": float(max_dimension),
            "max_dimension_mm": float(max_dimension * 1000),
            
            # Time histories
            "time_history": time_array.tolist(),
            "displacement_history": [d.tolist() for d in displacements],  # Relative (internal deformation)
            "absolute_displacement_history": [d.tolist() for d in absolute_displacements],  # Absolute (total movement)
            "velocity_history": [v.tolist() for v in velocities],
            "absolute_velocity_history": [v.tolist() for v in absolute_velocities],
            "acceleration_history": [a.tolist() for a in accelerations],
            "absolute_acceleration_history": [a.tolist() for a in absolute_accelerations],
            "ground_displacement_history": ground_displacements.tolist(),
            "damage_history": [d.tolist() for d in damages],
            "stress_history": stresses.tolist(),
            "base_shear_history": base_shears.tolist(),
            "kinetic_energy_history": kinetic_energies.tolist(),
        }
        
        return results


def run_earthquake_test(stl_path: Path, ground_motion: GroundMotion,
                       material: Optional[MaterialProperties] = None,
                       element_size: float = 0.005,
                       damping_ratio: float = 0.05) -> Dict:
    """Run earthquake simulation on an STL structure"""
    if material is None:
        material = MaterialProperties()
    
    print(f"Loading STL: {stl_path}")
    domain = load_stl_and_create_mesh(stl_path, element_size)
    
    sim_params = EarthquakeSimulationParameters(
        ground_motion=ground_motion,
        damping_ratio=damping_ratio,
        element_size=element_size
    )
    
    simulator = EarthquakeSimulator(domain, material, sim_params)
    results = simulator.run_simulation()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics-based earthquake simulator")
    parser.add_argument("stl_file", type=str, help="Path to STL file")
    parser.add_argument("--pga", type=float, help="Peak ground acceleration (m/s²)")
    parser.add_argument("--pga-g", type=float, help="Peak ground acceleration (g units)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration (seconds)")
    parser.add_argument("--damping", type=float, default=0.05, help="Damping ratio")
    parser.add_argument("--element-size", type=float, default=0.005, help="Mesh element size (m)")
    parser.add_argument("--output", type=str, default="earthquake_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Create intensity
    if args.pga:
        target_pga = args.pga
    elif args.pga_g:
        target_pga = args.pga_g * 9.81
    else:
        target_pga = 0.5 * 9.81  # Default 0.5g
    
    print(f"Generating synthetic ground motion: PGA = {target_pga:.3f} m/s² ({target_pga/9.81:.3f} g)")
    ground_motion = GroundMotion.synthetic(
        duration=args.duration,
        pga=target_pga,
        name=f"synthetic_{target_pga/9.81:.2f}g"
    )
    
    # Run simulation
    results = run_earthquake_test(
        Path(args.stl_file),
        ground_motion,
        element_size=args.element_size,
        damping_ratio=args.damping
    )
    
    # Add full ground motion history
    results['ground_motion_full_time'] = ground_motion.time.tolist()
    results['ground_motion_full_acceleration'] = ground_motion.acceleration.tolist()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Ground Motion: {results['ground_motion_name']}")
    print(f"PGA: {results['pga_g']:.3f} g ({results['pga']:.2f} m/s²)")
    print(f"\nMass & Weight:")
    print(f"  Total mass: {results['total_mass_kg']:.3f} kg")
    print(f"  Total weight: {results['total_weight_kN']:.2f} kN")
    print(f"  Natural frequency: {results['natural_frequency_Hz']:.3f} Hz")
    print(f"\nBase Shear (V = C × W):")
    print(f"  Max base shear: {results['max_base_shear_kN']:.2f} kN")
    print(f"  Base shear coefficient (C): {results['base_shear_coefficient']:.3f}")
    print(f"\nDisplacement - Relative (Internal Deformation):")
    print(f"  Max: {results['max_displacement_relative_mm']:.4f} mm")
    print(f"  Residual: {results['residual_displacement_relative_mm']:.4f} mm")
    print(f"\nDisplacement - Absolute (Total Movement):")
    print(f"  Max: {results['max_displacement_absolute_mm']:.2f} mm")
    print(f"  Residual: {results['residual_displacement_absolute_mm']:.2f} mm")
    print(f"  Ground motion contribution: {results['max_ground_displacement_mm']:.2f} mm")
    print(f"\nAcceleration:")
    print(f"  Peak relative: {results['peak_acceleration_relative_g']:.3f} g")
    print(f"  Peak absolute: {results['peak_acceleration_absolute_g']:.3f} g (should match PGA: {results['pga_g']:.3f} g)")
    print(f"\nStress (σ = F/A = ma/A):")
    print(f"  Max: {results['max_stress_MPa']:.6f} MPa")
    print(f"  Residual: {results['residual_stress_MPa']:.6f} MPa")
    print(f"\nKinetic Energy (KE = ½ × m × v²):")
    print(f"  Max: {results['max_kinetic_energy_J']:.6f} J")
    print(f"\nDamage:")
    print(f"  Max: {results['max_damage']:.3f}")
    print(f"  Residual: {results['residual_damage']:.3f}")
    print(f"\nResults saved to: {args.output}")
    print("="*60)

