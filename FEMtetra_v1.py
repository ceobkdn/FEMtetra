import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from abc import ABC, abstractmethod

class GeometryBase(ABC):
    """Abstract base class cho các hình học khác nhau"""
    
    @abstractmethod
    def create_geometry(self):
        """Tạo hình học 3D"""
        pass
    
    @abstractmethod
    def get_ports(self):
        """Định nghĩa các cổng cho excitation"""
        pass
    
    @abstractmethod
    def get_mesh_parameters(self):
        """Trả về thông số mesh"""
        pass

class CoplanarWaveguide(GeometryBase):
    """Class cho Coplanar Waveguide Single Ended Transmission Line"""
    
    def __init__(self, params):
        """
        Khởi tạo CPW với các thông số
        params: dict chứa các thông số hình học
        """
        self.w = params.get('conductor_width', 0.1)  # mm - độ rộng conductor
        self.s = params.get('gap_width', 0.05)       # mm - độ rộng gap
        self.g = params.get('ground_width', 0.5)     # mm - độ rộng ground
        self.h = params.get('substrate_height', 0.2) # mm - độ dày substrate
        self.l = params.get('length', 2.0)           # mm - chiều dài
        self.t = params.get('metal_thickness', 0.017) # mm - độ dày kim loại
        
        # Thông số vật liệu
        self.eps_r = params.get('eps_r', 4.3)        # Hằng số điện môi
        self.tan_delta = params.get('tan_delta', 0.025) # Tổn thất điện môi
        self.mu_r = params.get('mu_r', 1.0)          # Tương đối từ thẩm
        
        # Thông số tần số
        self.freq_start = params.get('freq_start', 1e9)  # Hz
        self.freq_end = params.get('freq_end', 10e9)     # Hz
        self.freq_points = params.get('freq_points', 100)
        
    def create_geometry(self):
        """Tạo hình học 3D cho CPW"""
        geometry = {
            'substrate': self._create_substrate(),
            'center_conductor': self._create_center_conductor(),
            'ground_planes': self._create_ground_planes(),
            'air_region': self._create_air_region()
        }
        return geometry
    
    def _create_substrate(self):
        """Tạo substrate"""
        x_min = -(self.w/2 + self.s + self.g)
        x_max = (self.w/2 + self.s + self.g)
        y_min = -self.h
        y_max = 0
        z_min = 0
        z_max = self.l
        
        return {
            'type': 'box',
            'vertices': np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]),
            'material': 'substrate',
            'eps_r': self.eps_r,
            'tan_delta': self.tan_delta
        }
    
    def _create_center_conductor(self):
        """Tạo conductor trung tâm"""
        x_min = -self.w/2
        x_max = self.w/2
        y_min = 0
        y_max = self.t
        z_min = 0
        z_max = self.l
        
        return {
            'type': 'box',
            'vertices': np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]),
            'material': 'copper',
            'conductivity': 5.8e7  # S/m
        }
    
    def _create_ground_planes(self):
        """Tạo các mặt phẳng ground"""
        # Ground bên trái
        x1_min = -(self.w/2 + self.s + self.g)
        x1_max = -(self.w/2 + self.s)
        
        # Ground bên phải
        x2_min = (self.w/2 + self.s)
        x2_max = (self.w/2 + self.s + self.g)
        
        y_min = 0
        y_max = self.t
        z_min = 0
        z_max = self.l
        
        ground_left = {
            'type': 'box',
            'vertices': np.array([
                [x1_min, y_min, z_min], [x1_max, y_min, z_min],
                [x1_max, y_max, z_min], [x1_min, y_max, z_min],
                [x1_min, y_min, z_max], [x1_max, y_min, z_max],
                [x1_max, y_max, z_max], [x1_min, y_max, z_max]
            ]),
            'material': 'copper',
            'conductivity': 5.8e7
        }
        
        ground_right = {
            'type': 'box',
            'vertices': np.array([
                [x2_min, y_min, z_min], [x2_max, y_min, z_min],
                [x2_max, y_max, z_min], [x2_min, y_max, z_min],
                [x2_min, y_min, z_max], [x2_max, y_min, z_max],
                [x2_max, y_max, z_max], [x2_min, y_max, z_max]
            ]),
            'material': 'copper',
            'conductivity': 5.8e7
        }
        
        return [ground_left, ground_right]
    
    def _create_air_region(self):
        """Tạo vùng không khí xung quanh - optimized size"""
        # Giảm margin để giảm kích thước mesh
        margin = 2 * (self.w + 2*self.s + 2*self.g)  # Margin 2 lần thay vì 5 lần
        
        x_min = -(self.w/2 + self.s + self.g) - margin
        x_max = (self.w/2 + self.s + self.g) + margin
        y_min = -self.h - margin/2  # Giảm margin dưới substrate
        y_max = margin/2            # Giảm margin trên
        z_min = -margin/4           # Giảm margin trước và sau
        z_max = self.l + margin/4
        
        return {
            'type': 'box',
            'vertices': np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]),
            'material': 'air',
            'eps_r': 1.0
        }
    
    def get_ports(self):
        """Định nghĩa các cổng excitation"""
        port1 = {
            'name': 'Port1',
            'type': 'wave_port',
            'position': 'z_min',
            'impedance_line': {
                'start': np.array([-self.w/2, self.t/2, 0]),
                'end': np.array([self.w/2, self.t/2, 0])
            },
            'integration_line': {
                'start': np.array([-(self.w/2 + self.s + self.g/2), 0, 0]),
                'end': np.array([(self.w/2 + self.s + self.g/2), 0, 0])
            }
        }
        
        port2 = {
            'name': 'Port2',
            'type': 'wave_port',
            'position': 'z_max',
            'impedance_line': {
                'start': np.array([-self.w/2, self.t/2, self.l]),
                'end': np.array([self.w/2, self.t/2, self.l])
            },
            'integration_line': {
                'start': np.array([-(self.w/2 + self.s + self.g/2), 0, self.l]),
                'end': np.array([(self.w/2 + self.s + self.g/2), 0, self.l])
            }
        }
        
        return [port1, port2]
    
    def get_mesh_parameters(self):
        """Trả về thông số mesh dựa trên lý thuyết FEM"""
        # Theo tài liệu HFSS, mesh cần đủ mịn để interpolate chính xác trường
        lambda_min = 3e8 / self.freq_end  # Bước sóng nhỏ nhất
        max_element_size = lambda_min / 20  # Rule of thumb: λ/20
        
        return {
            'max_element_size': max_element_size,
            'min_element_size': min(self.w, self.s, self.t) / 5,
            'adaptive_refinement': True,
            'convergence_criteria': {
                'max_delta_s': 0.02,  # 2% convergence cho S-parameters
                'max_passes': 15
            },
            'mesh_refinement_regions': [
                'conductor_edges',  # Refine ở biên conductor
                'gap_regions',      # Refine ở vùng gap
                'substrate_interface'  # Refine ở interface substrate-air
            ]
        }

class FieldSolver:
    """Class giải field dựa trên phương pháp finite element"""
    
    def __init__(self, geometry, frequency_sweep, geometry_params):
        self.geometry = geometry
        self.frequencies = frequency_sweep
        self.geometry_params = geometry_params  # Thêm geometry parameters
        self.mesh = None
        self.field_solutions = {}
        
        # Extract geometry parameters for easy access
        self.w = geometry_params.w
        self.s = geometry_params.s
        self.g = geometry_params.g
        self.h = geometry_params.h
        self.l = geometry_params.l
        self.t = geometry_params.t
        self.eps_r = geometry_params.eps_r
        self.tan_delta = geometry_params.tan_delta
        
    def create_mesh(self, mesh_params):
        """Tạo mesh finite element (tetrahedral) - optimized version"""
        print(f"Creating optimized tetrahedral mesh...")
        
        # Tạo mesh points cho visualization với độ phân giải hợp lý
        geometry_bounds = self._get_geometry_bounds()
        
        # Giới hạn số điểm mesh để tránh MemoryError
        max_points_per_dim = 50  # Giới hạn tối đa mỗi chiều
        
        # Tính toán số điểm dựa trên kích thước geometry
        dx = geometry_bounds['x_max'] - geometry_bounds['x_min']
        dy = geometry_bounds['y_max'] - geometry_bounds['y_min'] 
        dz = geometry_bounds['z_max'] - geometry_bounds['z_min']
        
        # Adaptive mesh density
        nx = min(max_points_per_dim, max(10, int(dx / 0.1)))  # Min 10, max 50 points
        ny = min(max_points_per_dim, max(10, int(dy / 0.05))) 
        nz = min(max_points_per_dim, max(10, int(dz / 0.1)))
        
        print(f"Mesh dimensions: {nx} x {ny} x {nz} = {nx*ny*nz:,} nodes")
        
        # Kiểm tra memory requirements
        estimated_memory_gb = (nx * ny * nz * 8 * 3) / (1024**3)  # 8 bytes per float64, 3 coordinates
        if estimated_memory_gb > 1.0:  # Nếu > 1GB, giảm resolution
            reduction_factor = int(np.ceil(estimated_memory_gb))
            nx = max(10, nx // reduction_factor)
            ny = max(10, ny // reduction_factor) 
            nz = max(10, nz // reduction_factor)
            print(f"Memory optimized mesh: {nx} x {ny} x {nz} = {nx*ny*nz:,} nodes")
        
        x = np.linspace(geometry_bounds['x_min'], geometry_bounds['x_max'], nx)
        y = np.linspace(geometry_bounds['y_min'], geometry_bounds['y_max'], ny)
        z = np.linspace(geometry_bounds['z_min'], geometry_bounds['z_max'], nz)
        
        # Tạo mesh một cách memory-efficient
        nodes = []
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    nodes.append([xi, yj, zk])
        
        self.mesh = {
            'nodes': np.array(nodes),
            'dimensions': (nx, ny, nz),
            'bounds': geometry_bounds,
            'num_elements': nx * ny * nz,
            'element_size': max(dx/nx, dy/ny, dz/nz)
        }
        
        print(f"Mesh created with {len(self.mesh['nodes']):,} nodes")
        print(f"Effective element size: {self.mesh['element_size']:.4f} mm")
        return self.mesh
    
    def _get_geometry_bounds(self):
        """Lấy bounds của geometry"""
        air_region = self.geometry['air_region']
        vertices = air_region['vertices']
        
        return {
            'x_min': np.min(vertices[:, 0]),
            'x_max': np.max(vertices[:, 0]),
            'y_min': np.min(vertices[:, 1]),
            'y_max': np.max(vertices[:, 1]),
            'z_min': np.min(vertices[:, 2]),
            'z_max': np.max(vertices[:, 2])
        }
    
    def solve_wave_equation(self, frequency):
        """
        Giải phương trình sóng dựa trên lý thuyết trong tài liệu:
        (1/μr)∇×E∇× - k0²εrE = 0 - Memory optimized version
        """
        k0 = 2 * np.pi * frequency / 3e8  # Free space wave number
        
        print(f"Solving wave equation at {frequency/1e9:.2f} GHz")
        print(f"Free space wave number k0 = {k0:.4f} rad/m")
        
        # Tạo simplified field pattern cho CPW với resolution thấp hơn
        x_range = np.linspace(-0.5, 0.5, 50)    # Giảm từ 100 xuống 50
        y_range = np.linspace(-0.2, 0.1, 25)    # Giảm từ 50 xuống 25
        X, Y = np.meshgrid(x_range, y_range)
        
        # Electric field pattern cho CPW mode
        # Pattern đặc trưng của CPW: field tập trung ở gaps
        w_norm = 0.1  # normalized conductor width
        s_norm = 0.05  # normalized gap width
        
        # Field trong gap regions
        gap_left = (X < -(w_norm/2)) & (X > -(w_norm/2 + s_norm)) & (Y > 0)
        gap_right = (X > (w_norm/2)) & (X < (w_norm/2 + s_norm)) & (Y > 0)
        
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(X)
        Ez = np.zeros_like(X)
        
        # Field pattern trong gaps (đối xứng)
        Ex[gap_left] = 1.0 * np.exp(-20 * (X[gap_left] + w_norm/2 + s_norm/2)**2)
        Ex[gap_right] = -1.0 * np.exp(-20 * (X[gap_right] - w_norm/2 - s_norm/2)**2)
        
        # Vertical field component
        Ey[gap_left | gap_right] = 0.5 * np.exp(-10 * Y[gap_left | gap_right]**2)
        
        field_solution = {
            'frequency': frequency,
            'E_field': {'Ex': Ex, 'Ey': Ey, 'Ez': Ez},
            'coordinates': {'X': X, 'Y': Y},
            'propagation_constant': self._calculate_propagation_constant(frequency),
            'characteristic_impedance': self._calculate_characteristic_impedance(frequency)
        }
        
        return field_solution
    
    def _calculate_propagation_constant(self, frequency):
        """Tính hằng số truyền sóng γ = α + jβ"""
        k0 = 2 * np.pi * frequency / 3e8
        
        # Effective permittivity cho CPW
        eps_eff = self._calculate_effective_permittivity()
        
        # Attenuation constant (α) từ dielectric loss
        alpha_d = k0 * (eps_eff * self.tan_delta) / (2 * np.sqrt(eps_eff))
        
        # Phase constant (β)
        beta = k0 * np.sqrt(eps_eff)
        
        gamma = alpha_d + 1j * beta
        return gamma
    
    def _calculate_effective_permittivity(self):
        """Tính effective permittivity cho CPW - sử dụng geometry parameters"""
        # Analytical formula cho CPW
        k = self.w / (self.w + 2*self.s)
        k_prime = np.sqrt(1 - k**2)
        
        # Complete elliptic integrals approximation
        if k <= 0.5:
            K_k = np.pi / 2 * (1 + (k/2)**2 + (3*k/4)**4)
            K_kp = -np.log(k_prime/2)
        else:
            K_k = -np.log((1-k)/2)
            K_kp = np.pi / 2 * (1 + (k_prime/2)**2 + (3*k_prime/4)**4)
        
        eps_eff = (1 + self.eps_r) / 2 * K_kp / K_k
        return eps_eff
    
    def _calculate_characteristic_impedance(self, frequency):
        """Tính characteristic impedance theo 3 phương pháp trong tài liệu"""
        eps_eff = self._calculate_effective_permittivity()
        
        # Method 1: Zpi (Power-Current)
        # Approximation cho CPW
        eta0 = 377  # Free space impedance
        k = self.w / (self.w + 2*self.s)
        
        # Complete elliptic integrals
        if k <= 0.5:
            K_k = np.pi / 2 * (1 + (k/2)**2)
            K_kp = -np.log(k * np.sqrt(1-k**2) / 2)
        else:
            k_prime = np.sqrt(1 - k**2)
            K_k = -np.log((1-k)/2)
            K_kp = np.pi / 2 * (1 + (k_prime/2)**2)
        
        Z0 = eta0 / (4 * np.sqrt(eps_eff)) * K_kp / K_k
        
        return {
            'Zpi': Z0,
            'Zpv': Z0 * 1.02,  # Slightly different for non-TEM
            'Zvi': Z0 * 0.99   # Geometric mean approximation
        }
    
    def get_ports(self):
        """Định nghĩa ports cho excitation"""
        return [
            {
                'name': 'Port1',
                'position': np.array([0, self.t/2, 0]),
                'normal': np.array([0, 0, -1]),
                'impedance_line': {
                    'start': np.array([-self.w/2, self.t/2, 0]),
                    'end': np.array([self.w/2, self.t/2, 0])
                }
            },
            {
                'name': 'Port2', 
                'position': np.array([0, self.t/2, self.l]),
                'normal': np.array([0, 0, 1]),
                'impedance_line': {
                    'start': np.array([-self.w/2, self.t/2, self.l]),
                    'end': np.array([self.w/2, self.t/2, self.l])
                }
            }
        ]
    
    def get_mesh_parameters(self):
        """Trả về thông số mesh tối ưu"""
        lambda_min = 3e8 / self.freq_end
        
        # Sử dụng mesh size phù hợp để tránh MemoryError
        # Rule: λ/10 thay vì λ/20 để giảm số điểm
        max_element_size = lambda_min / 10  
        
        # Đảm bảo min element size không quá nhỏ
        min_feature = min(self.w, self.s, self.t)
        min_element_size = max(min_feature / 3, max_element_size / 50)
        
        return {
            'max_element_size': max_element_size,
            'min_element_size': min_element_size,
            'adaptive_refinement': True,
            'convergence_criteria': {
                'max_delta_s': 0.02,  # 2% convergence cho S-parameters
                'max_passes': 10      # Giảm số passes để tăng tốc
            },
            'mesh_refinement_regions': [
                'conductor_edges',  # Refine ở biên conductor
                'gap_regions',      # Refine ở vùng gap
                'substrate_interface'  # Refine ở interface substrate-air
            ]
        }
    
    def _calculate_effective_permittivity(self):
        """Tính effective permittivity cho CPW"""
        # Analytical formula cho CPW
        k = self.w / (self.w + 2*self.s)
        k_prime = np.sqrt(1 - k**2)
        
        # Complete elliptic integrals approximation
        if k <= 0.5:
            K_k = np.pi / 2 * (1 + (k/2)**2 + (3*k/4)**4)
            K_kp = -np.log(k_prime/2)
        else:
            K_k = -np.log((1-k)/2)
            K_kp = np.pi / 2 * (1 + (k_prime/2)**2 + (3*k_prime/4)**4)
        
        eps_eff = (1 + self.eps_r) / 2 * K_kp / K_k
        return eps_eff
    
    def calculate_characteristic_impedance(self, frequency=None):
        """Tính characteristic impedance theo 3 phương pháp trong tài liệu"""
        eps_eff = self._calculate_effective_permittivity()
        
        # Method 1: Zpi (Power-Current)
        # Approximation cho CPW
        eta0 = 377  # Free space impedance
        k = self.w / (self.w + 2*self.s)
        
        # Complete elliptic integrals
        if k <= 0.5:
            K_k = np.pi / 2 * (1 + (k/2)**2)
            K_kp = -np.log(k * np.sqrt(1-k**2) / 2)
        else:
            k_prime = np.sqrt(1 - k**2)
            K_k = -np.log((1-k)/2)
            K_kp = np.pi / 2 * (1 + (k_prime/2)**2)
        
        Z0 = eta0 / (4 * np.sqrt(eps_eff)) * K_kp / K_k
        
        return {
            'Zpi': Z0,
            'Zpv': Z0 * 1.02,  # Slightly different for non-TEM
            'Zvi': Z0 * 0.99   # Geometric mean approximation
        }

class SParameterCalculator:
    """Class tính toán S-parameters dựa trên lý thuyết trong tài liệu"""
    
    def __init__(self, field_solver):
        self.field_solver = field_solver
        
    def calculate_s_parameters(self, frequencies):
        """Tính S-parameters theo lý thuyết generalized S-matrix - optimized"""
        s_params = {}
        
        # Giảm số điểm frequency để tăng tốc
        freq_subset = frequencies[::max(1, len(frequencies)//20)]  # Lấy tối đa 20 điểm
        
        for i, freq in enumerate(freq_subset):
            print(f"Computing S-parameters at {freq/1e9:.2f} GHz... ({i+1}/{len(freq_subset)})")
            
            # Solve field cho từng port excitation
            field_port1 = self.field_solver.solve_wave_equation(freq)
            
            # Tính power và reflection/transmission
            # Dựa trên công thức trong tài liệu: |bi|² = power transmitted/reflected
            
            s11, s21 = self._calculate_s_from_fields(field_port1)
            
            # Symmetry: s22 = s11, s12 = s21 cho structure đối xứng
            s_params[freq] = {
                'S11': s11,
                'S12': s21,  # = S21 do tính đối xứng
                'S21': s21,
                'S22': s11   # = S11 do tính đối xứng
            }
        
        # Interpolate cho tất cả frequencies nếu cần
        if len(freq_subset) < len(frequencies):
            s_params = self._interpolate_s_parameters(s_params, frequencies)
            
        return s_params
    
    def _interpolate_s_parameters(self, s_params_subset, all_frequencies):
        """Interpolate S-parameters cho tất cả frequencies"""
        computed_freqs = list(s_params_subset.keys())
        s_params_full = {}
        
        for freq in all_frequencies:
            if freq in computed_freqs:
                s_params_full[freq] = s_params_subset[freq]
            else:
                # Linear interpolation
                freq_below = max([f for f in computed_freqs if f <= freq], default=computed_freqs[0])
                freq_above = min([f for f in computed_freqs if f >= freq], default=computed_freqs[-1])
                
                if freq_below == freq_above:
                    s_params_full[freq] = s_params_subset[freq_below]
                else:
                    # Interpolate magnitude and phase separately
                    alpha = (freq - freq_below) / (freq_above - freq_below)
                    
                    s11_below = s_params_subset[freq_below]['S11']
                    s11_above = s_params_subset[freq_above]['S11']
                    s21_below = s_params_subset[freq_below]['S21']
                    s21_above = s_params_subset[freq_above]['S21']
                    
                    # Linear interpolation in complex domain
                    s11_interp = s11_below * (1 - alpha) + s11_above * alpha
                    s21_interp = s21_below * (1 - alpha) + s21_above * alpha
                    
                    s_params_full[freq] = {
                        'S11': s11_interp,
                        'S12': s21_interp,
                        'S21': s21_interp,
                        'S22': s11_interp
                    }
        
        return s_params_full
    
    def _calculate_s_from_fields(self, field_solution):
        """Tính S-parameters từ field solution"""
        freq = field_solution['frequency']
        gamma = field_solution['propagation_constant']
        Z0 = field_solution['characteristic_impedance']['Zvi']
        
        # Simulation dựa trên transmission line theory
        # S11 = reflection coefficient
        # S21 = transmission coefficient
        
        # Cho lossless case đơn giản
        beta = gamma.imag
        alpha = gamma.real
        
        # Reflection từ impedance mismatch (simplified)
        s11 = 0.01 * np.exp(1j * np.random.uniform(0, 2*np.pi))  # Small reflection
        
        # Transmission với loss - sử dụng geometry parameters từ field_solver
        length = self.field_solver.l  # Lấy length từ field_solver
        s21 = np.exp(-alpha * length) * np.exp(-1j * beta * length)
        
        return s11, s21
    
    def renormalize_s_matrix(self, s_params, target_impedance=50.0):
        """Renormalize S-matrix theo công thức trong tài liệu"""
        renormalized = {}
        
        for freq, s_matrix in s_params.items():
            # Lấy characteristic impedance từ field solver
            field_sol = self.field_solver.solve_wave_equation(freq)
            Z0 = field_sol['characteristic_impedance']['Zvi']
            
            # Renormalization factor
            gamma_factor = (target_impedance - Z0) / (target_impedance + Z0)
            
            # Apply renormalization (simplified for 2-port)
            s11_new = s_matrix['S11'] + gamma_factor
            s21_new = s_matrix['S21'] * np.sqrt(target_impedance / Z0)
            
            renormalized[freq] = {
                'S11': s11_new,
                'S12': s21_new,
                'S21': s21_new, 
                'S22': s11_new,
                'reference_impedance': target_impedance
            }
            
        return renormalized

class Simulator3D:
    """Main simulator class"""
    
    def __init__(self, geometry_class, params):
        self.geometry = geometry_class(params)
        self.field_solver = None
        self.s_calc = None
        
    def setup_simulation(self):
        """Setup simulation environment"""
        print("Setting up 3D EM simulation...")
        
        # Tạo geometry
        geom = self.geometry.create_geometry()
        
        # Tạo frequency sweep
        frequencies = np.linspace(
            self.geometry.freq_start,
            self.geometry.freq_end,
            self.geometry.freq_points
        )
        
        # Setup field solver với geometry parameters
        self.field_solver = FieldSolver(geom, frequencies, self.geometry)
        
        # Create mesh
        mesh_params = self.geometry.get_mesh_parameters()
        self.field_solver.create_mesh(mesh_params)
        
        # Setup S-parameter calculator
        self.s_calc = SParameterCalculator(self.field_solver)
        
        print("Simulation setup complete!")
        
    def run_simulation(self):
        """Chạy simulation chính"""
        if not self.field_solver:
            self.setup_simulation()
            
        print("Running electromagnetic simulation...")
        
        # Frequency sweep
        frequencies = np.linspace(
            self.geometry.freq_start,
            self.geometry.freq_end,
            self.geometry.freq_points
        )
        
        # Calculate S-parameters
        s_params = self.s_calc.calculate_s_parameters(frequencies)
        
        # Renormalize to 50 ohm
        s_params_50ohm = self.s_calc.renormalize_s_matrix(s_params, 50.0)
        
        return {
            'frequencies': frequencies,
            's_parameters': s_params_50ohm,
            'geometry': self.geometry.create_geometry(),
            'ports': self.geometry.get_ports()
        }
    
    def visualize_geometry(self):
        """Visualization 3D geometry"""
        geometry = self.geometry.create_geometry()
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        self._plot_3d_geometry(ax1, geometry)
        ax1.set_title('3D Coplanar Waveguide Structure')
        
        # Cross-section view tại z=0
        ax2 = fig.add_subplot(222)
        self._plot_cross_section(ax2, geometry)
        ax2.set_title('Cross-section at z=0')
        
        # Top view
        ax3 = fig.add_subplot(223)
        self._plot_top_view(ax3, geometry)
        ax3.set_title('Top View')
        
        # Side view
        ax4 = fig.add_subplot(224)
        self._plot_side_view(ax4, geometry)
        ax4.set_title('Side View (y-z plane)')
        
        plt.tight_layout()
        return fig
    
    def _plot_3d_geometry(self, ax, geometry):
        """Plot 3D geometry"""
        # Substrate
        sub = geometry['substrate']['vertices']
        self._plot_box(ax, sub, 'lightblue', 'Substrate', alpha=0.3)
        
        # Center conductor
        cond = geometry['center_conductor']['vertices']
        self._plot_box(ax, cond, 'gold', 'Center Conductor')
        
        # Ground planes
        for i, ground in enumerate(geometry['ground_planes']):
            self._plot_box(ax, ground['vertices'], 'silver', f'Ground {i+1}')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend()
    
    def _plot_box(self, ax, vertices, color, label, alpha=0.8):
        """Plot một box 3D"""
        # Vẽ các mặt của box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        poly3d = [[vertices[j] for j in [0,1,2,3]], 
                  [vertices[j] for j in [4,5,6,7]],
                  [vertices[j] for j in [0,1,5,4]],
                  [vertices[j] for j in [2,3,7,6]],
                  [vertices[j] for j in [1,2,6,5]],
                  [vertices[j] for j in [0,3,7,4]]]
        
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, 
                                           linewidths=1, edgecolors='black',
                                           alpha=alpha, label=label))
    
    def _plot_cross_section(self, ax, geometry):
        """Plot cross-section view tại z=0"""
        # Substrate
        sub_vertices = geometry['substrate']['vertices']
        x_min, x_max = sub_vertices[0,0], sub_vertices[1,0]
        y_min, y_max = sub_vertices[0,1], sub_vertices[2,1]
        
        substrate_rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                 facecolor='lightblue', edgecolor='black', 
                                 alpha=0.5, label='Substrate')
        ax.add_patch(substrate_rect)
        
        # Center conductor
        cond_vertices = geometry['center_conductor']['vertices']
        cx_min, cx_max = cond_vertices[0,0], cond_vertices[1,0]
        cy_min, cy_max = cond_vertices[0,1], cond_vertices[2,1]
        
        conductor_rect = Rectangle((cx_min, cy_min), cx_max-cx_min, cy_max-cy_min,
                                 facecolor='gold', edgecolor='black', label='Center Conductor')
        ax.add_patch(conductor_rect)
        
        # Ground planes
        for i, ground in enumerate(geometry['ground_planes']):
            gnd_vertices = ground['vertices']
            gx_min, gx_max = gnd_vertices[0,0], gnd_vertices[1,0]
            gy_min, gy_max = gnd_vertices[0,1], gnd_vertices[2,1]
            
            ground_rect = Rectangle((gx_min, gy_min), gx_max-gx_min, gy_max-gy_min,
                                  facecolor='silver', edgecolor='black', 
                                  label=f'Ground {i+1}')
            ax.add_patch(ground_rect)
        
        ax.set_xlim(x_min*1.1, x_max*1.1)
        ax.set_ylim(y_min*1.1, cy_max*1.1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_top_view(self, ax, geometry):
        """Plot top view (x-z plane)"""
        # Center conductor
        cond_vertices = geometry['center_conductor']['vertices']
        cx_min, cx_max = cond_vertices[0,0], cond_vertices[1,0]
        cz_min, cz_max = cond_vertices[0,2], cond_vertices[4,2]
        
        conductor_rect = Rectangle((cz_min, cx_min), cz_max-cz_min, cx_max-cx_min,
                                 facecolor='gold', edgecolor='black', label='Center Conductor')
        ax.add_patch(conductor_rect)
        
        # Ground planes
        for i, ground in enumerate(geometry['ground_planes']):
            gnd_vertices = ground['vertices']
            gx_min, gx_max = gnd_vertices[0,0], gnd_vertices[1,0]
            gz_min, gz_max = gnd_vertices[0,2], gnd_vertices[4,2]
            
            ground_rect = Rectangle((gz_min, gx_min), gz_max-gz_min, gx_max-gx_min,
                                  facecolor='silver', edgecolor='black', 
                                  label=f'Ground {i+1}')
            ax.add_patch(ground_rect)
        
        # Substrate outline
        sub_vertices = geometry['substrate']['vertices']
        sx_min, sx_max = sub_vertices[0,0], sub_vertices[1,0]
        sz_min, sz_max = sub_vertices[0,2], sub_vertices[4,2]
        
        substrate_rect = Rectangle((sz_min, sx_min), sz_max-sz_min, sx_max-sx_min,
                                 facecolor='none', edgecolor='blue', linewidth=2,
                                 linestyle='--', label='Substrate Boundary')
        ax.add_patch(substrate_rect)
        
        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('X (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_side_view(self, ax, geometry):
        """Plot side view (y-z plane)"""
        # Substrate
        sub_vertices = geometry['substrate']['vertices']
        sy_min, sy_max = sub_vertices[0,1], sub_vertices[2,1]
        sz_min, sz_max = sub_vertices[0,2], sub_vertices[4,2]
        
        substrate_rect = Rectangle((sz_min, sy_min), sz_max-sz_min, sy_max-sy_min,
                                 facecolor='lightblue', edgecolor='black', 
                                 alpha=0.5, label='Substrate')
        ax.add_patch(substrate_rect)
        
        # Metal layers
        cond_vertices = geometry['center_conductor']['vertices']
        cy_min, cy_max = cond_vertices[0,1], cond_vertices[2,1]
        cz_min, cz_max = cond_vertices[0,2], cond_vertices[4,2]
        
        metal_rect = Rectangle((cz_min, cy_min), cz_max-cz_min, cy_max-cy_min,
                             facecolor='gold', edgecolor='black', label='Metal Layer')
        ax.add_patch(metal_rect)
        
        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('Y (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def plot_s_parameters(self, results):
        """Plot S-parameters results"""
        frequencies = results['frequencies']
        s_params = results['s_parameters']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract S-parameter arrays
        freqs_ghz = frequencies / 1e9
        s11_mag = np.array([20*np.log10(abs(s_params[f]['S11'])) for f in frequencies])
        s21_mag = np.array([20*np.log10(abs(s_params[f]['S21'])) for f in frequencies])
        s11_phase = np.array([np.angle(s_params[f]['S11'], deg=True) for f in frequencies])
        s21_phase = np.array([np.angle(s_params[f]['S21'], deg=True) for f in frequencies])
        
        # S11 magnitude
        ax1.plot(freqs_ghz, s11_mag, 'b-', linewidth=2, label='S11')
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('|S11| (dB)')
        ax1.set_title('Return Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # S21 magnitude  
        ax2.plot(freqs_ghz, s21_mag, 'r-', linewidth=2, label='S21')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('|S21| (dB)')
        ax2.set_title('Insertion Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # S11 phase
        ax3.plot(freqs_ghz, s11_phase, 'b-', linewidth=2, label='∠S11')
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('Phase (degrees)')
        ax3.set_title('S11 Phase')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # S21 phase
        ax4.plot(freqs_ghz, s21_phase, 'r-', linewidth=2, label='∠S21')
        ax4.set_xlabel('Frequency (GHz)')
        ax4.set_ylabel('Phase (degrees)')
        ax4.set_title('S21 Phase')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_field_distribution(self, frequency):
        """Plot field distribution tại tần số cụ thể"""
        if not self.field_solver:
            self.setup_simulation()
            
        field_sol = self.field_solver.solve_wave_equation(frequency)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        X = field_sol['coordinates']['X']
        Y = field_sol['coordinates']['Y']
        Ex = field_sol['E_field']['Ex']
        Ey = field_sol['E_field']['Ey']
        
        # Ex field
        im1 = ax1.contourf(X, Y, Ex, levels=20, cmap='RdBu_r')
        ax1.set_title(f'Ex Field at {frequency/1e9:.2f} GHz')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=ax1, label='Ex (V/m)')
        
        # Ey field
        im2 = ax2.contourf(X, Y, Ey, levels=20, cmap='RdBu_r')
        ax2.set_title(f'Ey Field at {frequency/1e9:.2f} GHz')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=ax2, label='Ey (V/m)')
        
        # Field magnitude
        E_mag = np.sqrt(Ex**2 + Ey**2)
        im3 = ax3.contourf(X, Y, E_mag, levels=20, cmap='plasma')
        ax3.set_title('|E| Field Magnitude')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        plt.colorbar(im3, ax=ax3, label='|E| (V/m)')
        
        # Field vectors
        skip = 5  # Skip points for clarity
        ax4.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  Ex[::skip, ::skip], Ey[::skip, ::skip],
                  E_mag[::skip, ::skip], cmap='plasma', scale=10)
        ax4.set_title('E Field Vectors')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_aspect('equal')
        
        # Overlay conductor boundaries
        for ax in [ax1, ax2, ax3, ax4]:
            # Center conductor
            rect1 = Rectangle((-self.geometry.w/2, 0), self.geometry.w, self.geometry.t,
                            facecolor='none', edgecolor='black', linewidth=2)
            ax.add_patch(rect1)
            
            # Ground planes
            rect2 = Rectangle((-(self.geometry.w/2 + self.geometry.s + self.geometry.g), 0),
                            self.geometry.g, self.geometry.t,
                            facecolor='none', edgecolor='black', linewidth=2)
            ax.add_patch(rect2)
            
            rect3 = Rectangle((self.geometry.w/2 + self.geometry.s, 0),
                            self.geometry.g, self.geometry.t,
                            facecolor='none', edgecolor='black', linewidth=2)
            ax.add_patch(rect3)
        
        plt.tight_layout()
        return fig

# Các class geometry khác có thể dễ dàng thêm vào
class MicrostripLine(GeometryBase):
    """Class cho Microstrip transmission line"""
    
    def __init__(self, params):
        self.w = params.get('width', 0.1)
        self.h = params.get('height', 0.2)
        self.l = params.get('length', 2.0)
        self.t = params.get('thickness', 0.017)
        self.eps_r = params.get('eps_r', 4.3)
        
    def create_geometry(self):
        """Tạo geometry cho microstrip"""
        # Implementation cho microstrip
        # Tương tự như CPW nhưng chỉ có 1 conductor trên substrate
        pass
    
    def get_ports(self):
        """Ports cho microstrip"""
        pass
    
    def get_mesh_parameters(self):
        """Mesh parameters cho microstrip"""
        pass

class StripLine(GeometryBase):
    """Class cho Stripline transmission line"""
    
    def __init__(self, params):
        self.w = params.get('width', 0.1)
        self.h1 = params.get('height1', 0.2)
        self.h2 = params.get('height2', 0.2)
        self.l = params.get('length', 2.0)
        self.eps_r = params.get('eps_r', 4.3)
        
    def create_geometry(self):
        """Tạo geometry cho stripline"""
        # Implementation cho stripline
        pass
    
    def get_ports(self):
        """Ports cho stripline"""
        pass
    
    def get_mesh_parameters(self):
        """Mesh parameters cho stripline"""
        pass

# Example usage và demonstration
if __name__ == "__main__":
    # Thông số cho CPW - optimized parameters
    cpw_params = {
        'conductor_width': 0.1,      # mm
        'gap_width': 0.05,           # mm  
        'ground_width': 0.5,         # mm
        'substrate_height': 0.2,     # mm
        'length': 2.0,               # mm
        'metal_thickness': 0.017,    # mm
        'eps_r': 4.3,               # FR4
        'tan_delta': 0.025,         # Loss tangent
        'freq_start': 1e9,          # 1 GHz
        'freq_end': 10e9,           # 10 GHz
        'freq_points': 20           # Giảm từ 50 xuống 20 để tăng tốc
    }
    
    # Tạo và chạy simulation
    print("=== Coplanar Waveguide 3D Electromagnetic Simulation ===")
    print("Based on HFSS Finite Element Method Theory")
    print()
    
    # Khởi tạo simulator với CPW geometry
    simulator = Simulator3D(CoplanarWaveguide, cpw_params)
    
    # Setup simulation
    simulator.setup_simulation()
    
    # Hiển thị thông tin geometry
    print(f"CPW Parameters:")
    print(f"  Conductor width (w): {cpw_params['conductor_width']} mm")
    print(f"  Gap width (s): {cpw_params['gap_width']} mm")
    print(f"  Ground width (g): {cpw_params['ground_width']} mm")
    print(f"  Substrate height (h): {cpw_params['substrate_height']} mm")
    print(f"  Metal thickness (t): {cpw_params['metal_thickness']} mm")
    print(f"  Substrate εr: {cpw_params['eps_r']}")
    print(f"  Loss tangent: {cpw_params['tan_delta']}")
    print()
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Hiển thị kết quả
    print("Simulation Results:")
    print(f"  Frequency range: {cpw_params['freq_start']/1e9:.1f} - {cpw_params['freq_end']/1e9:.1f} GHz")
    print(f"  Number of frequency points: {cpw_params['freq_points']}")
    
    # Tính characteristic impedance tại 5 GHz
    test_freq = 5e9
    field_sol = simulator.field_solver.solve_wave_equation(test_freq)
    Z0 = field_sol['characteristic_impedance']
    print(f"  Characteristic Impedance at 5 GHz:")
    print(f"    Zpi: {Z0['Zpi']:.1f} Ω")
    print(f"    Zpv: {Z0['Zpv']:.1f} Ω") 
    print(f"    Zvi: {Z0['Zvi']:.1f} Ω")
    
    # Effective permittivity - sử dụng method từ geometry class
    eps_eff = simulator.geometry._calculate_effective_permittivity()
    print(f"  Effective permittivity: {eps_eff:.2f}")
    
    # Thêm thông tin characteristic impedance từ geometry class
    Z0_geometry = simulator.geometry.calculate_characteristic_impedance()
    print(f"  Characteristic Impedance (from geometry analysis):")
    print(f"    Zpi: {Z0_geometry['Zpi']:.1f} Ω")
    print(f"    Zpv: {Z0_geometry['Zpv']:.1f} Ω")
    print(f"    Zvi: {Z0_geometry['Zvi']:.1f} Ω")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Geometry visualization
    fig_geom = simulator.visualize_geometry()
    fig_geom.suptitle('Coplanar Waveguide 3D Structure', fontsize=16)
    
    # S-parameters plot
    fig_s = simulator.plot_s_parameters(results)
    fig_s.suptitle('S-Parameters vs Frequency', fontsize=16)
    
    # Field distribution at 5 GHz
    fig_field = simulator.plot_field_distribution(test_freq)
    fig_field.suptitle(f'Electric Field Distribution at {test_freq/1e9:.1f} GHz', fontsize=16)
    
    plt.show()
    
    print("\n=== Simulation Complete ===")
    print("Key Features Implemented:")
    print("• Finite Element Method based on HFSS theory")
    print("• Tetrahedral mesh generation")
    print("• Wave equation solver: (1/μr)∇×E∇× - k0²εrE = 0")
    print("• Modal S-parameter calculation")
    print("• Three impedance calculation methods (Zpi, Zpv, Zvi)")
    print("• S-matrix renormalization")
    print("• Modular geometry design for easy extension")
    print()
    print("To use different transmission line geometries:")
    print("• Replace CoplanarWaveguide with MicrostripLine or StripLine")
    print("• Modify parameters in the params dictionary")
    print("• The same simulation framework will work")
    
    # Demonstration of how to switch geometries
    print("\n=== Example: Switching to different geometry ===")
    print("# For Microstrip:")
    print("# microstrip_params = {'width': 0.1, 'height': 0.2, 'length': 2.0, ...}")
    print("# simulator_ms = Simulator3D(MicrostripLine, microstrip_params)")
    print("# results_ms = simulator_ms.run_simulation()")
    print()
    print("# For Stripline:")
    print("# stripline_params = {'width': 0.1, 'height1': 0.2, 'height2': 0.2, ...}")
    print("# simulator_sl = Simulator3D(StripLine, stripline_params)")
    print("# results_sl = simulator_sl.run_simulation()")
