import numpy as np
from skimage import measure
from stl import mesh
import matplotlib.pyplot as plt

# Parameters
container_size = 20.0  # mm - size of rectangular container (20)
resolution = 80  # points per dimension (80)
max_iterations = 5  # fractal iteration depth (5) - more iterations = more branches
branch_probability = 0.9  # probability of branching (0.9) - higher = more branches
branch_angle = np.pi/8  # angle between branches (22.5 degrees) - smaller angle = denser packing
branch_length_factor = 0.7  # how much shorter each branch is (0.7) - longer branches = more overlap
branch_thickness_factor = 0.8  # how much thinner each branch is (0.8) - thicker branches = more volume
base_thickness = 0.8  # mm - thickness of main branches (0.8) - thinner = more branches fit
num_initial_branches = 5  # number of main branches from center (20) - more main branches for better distribution

# Create 3D grid
x = np.linspace(-container_size/2, container_size/2, resolution)
y = np.linspace(-container_size/2, container_size/2, resolution)
z = np.linspace(-container_size/2, container_size/2, resolution)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initialize the coral volume
coral_volume = np.zeros_like(X, dtype=bool)

def create_cylinder_segment(start_point, end_point, radius):
    """Create a cylindrical segment between two points"""
    # Vector along the cylinder axis
    axis = end_point - start_point
    length = np.linalg.norm(axis)
    if length < 0.1:  # Skip very short segments
        return
    
    axis = axis / length
    
    # Create points along the cylinder
    num_steps = max(5, int(length * 10))  # At least 5 steps, more for longer cylinders
    t_values = np.linspace(0, 1, num_steps)
    
    for t in t_values:
        # Current position along cylinder
        current_pos = start_point + t * axis * length
        
        # Calculate distance from this point to all grid points
        dx = X - current_pos[0]
        dy = Y - current_pos[1]
        dz = Z - current_pos[2]
        
        # Distance from cylinder axis
        dist_to_axis = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Mark points within the cylinder radius
        cylinder_mask = dist_to_axis <= radius
        coral_volume[cylinder_mask] = True

def create_branch(start_point, direction, length, thickness, iteration):
    """Recursively create a branch and its sub-branches"""
    if iteration >= max_iterations or length < 1.5:
        return
    
    # Add some natural curve to the branch direction
    curve_magnitude = 0.12  # Reduced from 0.15 for more even distribution
    curve_direction = np.array([np.random.uniform(-1, 1) for _ in range(3)])
    curve_direction = curve_direction / np.linalg.norm(curve_direction)
    
    # Create curved path for the branch
    num_segments = max(3, int(length * 4))
    segment_length = length / num_segments
    
    current_pos = start_point
    current_direction = direction
    
    # Track branch positions for better sub-branch distribution
    branch_positions = []
    
    for i in range(num_segments):
        # Calculate next position with curve
        next_pos = current_pos + current_direction * segment_length
        
        # Add curve to direction
        curve_effect = curve_direction * curve_magnitude * segment_length
        next_pos += curve_effect
        
        # Create cylinder segment
        create_cylinder_segment(current_pos, next_pos, thickness)
        
        # Store position for sub-branching
        branch_positions.append(current_pos.copy())
        
        # Update for next segment - check for zero length
        direction_vector = next_pos - current_pos
        direction_length = np.linalg.norm(direction_vector)
        
        if direction_length > 0.01:  # Only update if there's meaningful movement
            current_direction = direction_vector / direction_length
        
        current_pos = next_pos
    
    # Add final position
    branch_positions.append(current_pos.copy())
    
    # Decide whether to create sub-branches
    if iteration < max_iterations - 1 and np.random.random() < branch_probability:
        # Create 2-4 sub-branches with better distribution along the branch
        num_sub_branches = np.random.randint(2, 5)
        
        # Choose positions along the branch for sub-branches (avoid start and end)
        if len(branch_positions) > 2:
            # Use positions from middle sections for better distribution
            start_idx = max(1, len(branch_positions) // 4)
            end_idx = min(len(branch_positions) - 1, 3 * len(branch_positions) // 4)
            
            # Select evenly spaced positions for sub-branches
            if end_idx > start_idx:
                sub_branch_indices = np.linspace(start_idx, end_idx, num_sub_branches, dtype=int)
            else:
                sub_branch_indices = [len(branch_positions) // 2] * num_sub_branches
        else:
            sub_branch_indices = [len(branch_positions) // 2] * num_sub_branches
        
        for i in range(num_sub_branches):
            if i < len(sub_branch_indices):
                branch_start = branch_positions[sub_branch_indices[i]]
            else:
                branch_start = current_pos
            
            # Calculate new direction with improved distribution
            angle = branch_angle * (1 + np.random.uniform(-0.2, 0.2))  # Reduced randomness
            
            # Create rotation matrix around a random axis perpendicular to current direction
            perp_axis = np.array([np.random.uniform(-1, 1) for _ in range(3)])
            perp_axis = perp_axis - np.dot(perp_axis, current_direction) * current_direction
            
            # Check if perp_axis is not zero
            perp_norm = np.linalg.norm(perp_axis)
            if perp_norm > 0.01:
                perp_axis = perp_axis / perp_norm
                
                # Rotate direction
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                new_direction = (current_direction * cos_angle + 
                               np.cross(perp_axis, current_direction) * sin_angle +
                               perp_axis * np.dot(perp_axis, current_direction) * (1 - cos_angle))
                new_direction = new_direction / np.linalg.norm(new_direction)
                
                # Create sub-branch with better length distribution
                new_length = length * branch_length_factor * (0.7 + 0.6 * np.random.random())
                new_thickness = thickness * branch_thickness_factor
                
                create_branch(branch_start, new_direction, new_length, new_thickness, iteration + 1)

def generate_coral():
    """Generate a coral structure with multiple main branches using Fibonacci sphere distribution"""
    center = np.array([0, 0, 0])
    
    # Use Fibonacci sphere distribution for more even spacing
    def fibonacci_sphere(samples=num_initial_branches):
        """Generate evenly distributed points on a sphere using Fibonacci spiral"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    # Generate evenly distributed directions
    directions = fibonacci_sphere(num_initial_branches)
    
    # Create multiple main branches growing outward from center
    for i in range(num_initial_branches):
        # Get the pre-calculated direction
        direction = directions[i]
        
        # Add some randomness to make it more natural (but keep it small for even distribution)
        direction += np.random.uniform(-0.2, 0.2, 3)
        direction = direction / np.linalg.norm(direction)
        
        # Randomize branch length and thickness slightly
        branch_length = container_size * 0.45 * (0.7 + 0.6 * np.random.random())
        branch_thickness = base_thickness * (0.8 + 0.4 * np.random.random())
        
        print(f"Creating main branch {i+1}/{num_initial_branches}")
        create_branch(center, direction, branch_length, branch_thickness, 0)
    
    # Add additional branches from different starting points for better distribution
    num_additional_start_points = 8
    print(f"Creating {num_additional_start_points} additional starting points for better distribution...")
    
    for i in range(num_additional_start_points):
        # Create starting points distributed throughout the cube
        start_radius = container_size * 0.3 * (0.3 + 0.7 * np.random.random())
        start_phi = np.random.uniform(0, 2 * np.pi)
        start_theta = np.arccos(2 * np.random.random() - 1)
        
        start_x = start_radius * np.sin(start_theta) * np.cos(start_phi)
        start_y = start_radius * np.sin(start_theta) * np.sin(start_phi)
        start_z = start_radius * np.cos(start_theta)
        
        start_point = np.array([start_x, start_y, start_z])
        
        # Create 2-3 branches from each additional starting point
        num_branches_from_point = np.random.randint(2, 4)
        
        for j in range(num_branches_from_point):
            # Random direction from this point
            direction = np.array([np.random.uniform(-1, 1) for _ in range(3)])
            direction = direction / np.linalg.norm(direction)
            
            # Shorter branches from secondary points
            branch_length = container_size * 0.25 * (0.6 + 0.8 * np.random.random())
            branch_thickness = base_thickness * 0.7 * (0.7 + 0.6 * np.random.random())
            
            print(f"Creating additional branch {j+1}/{num_branches_from_point} from point {i+1}")
            create_branch(start_point, direction, branch_length, branch_thickness, 1)  # Start at iteration 1

# Generate the coral structure

print("Generating coral structure...")
generate_coral()

# Add a solid shell (face) around the perimeter of the cube
shell_thickness = max(container_size / resolution, 0.5)  # At least one voxel thick
outer_shell_mask = (
    (np.abs(X) >= (container_size/2 - shell_thickness)) |
    (np.abs(Y) >= (container_size/2 - shell_thickness)) |
    (np.abs(Z) >= (container_size/2 - shell_thickness))
)

# Merge the shell with the coral structure
coral_with_shell = coral_volume | outer_shell_mask

# Apply container bounds (rectangular container)

container_mask = (X >= -container_size/2) & (X <= container_size/2) & \
                 (Y >= -container_size/2) & (Y <= container_size/2) & \
                 (Z >= -container_size/2) & (Z <= container_size/2)

# Use the coral structure with the shell
final_volume = coral_with_shell & container_mask

# Remove small disconnected components: keep only the largest connected component
labels = measure.label(final_volume.astype(np.uint8), connectivity=1)
if labels.max() > 0:
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0  # ignore background
    main_label = component_sizes.argmax()
    cleaned_volume = labels == main_label
    removed_voxels = int(final_volume.sum() - cleaned_volume.sum())
    num_components = int((component_sizes > 0).sum())
    print(f"Removed disconnected components: kept 1 of {num_components} components; removed {removed_voxels} voxels")
    final_volume = cleaned_volume
else:
    print("No components found (empty volume). Proceeding without cleanup.")

# Marching cubes to create mesh
print("Creating mesh with marching cubes...")
verts, faces, normals, values = measure.marching_cubes(
    final_volume.astype(float), 
    0.5, 
    spacing=(container_size/resolution, container_size/resolution, container_size/resolution)
)

# Create STL mesh
coral_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        coral_mesh.vectors[i][j] = verts[f[j], :]

# Verify solid voxels
print(f"Total solid voxels: {np.sum(final_volume)}")

# Save STL file
coral_mesh.save('_coral_v2.stl')
print(f"Saved _coral_v2.stl with {len(faces)} faces")

# Optional: Create a simple visualization
print("Creating visualization...")
fig = plt.figure(figsize=(12, 8))

# Show middle slice
mid_z = resolution // 2
ax1 = fig.add_subplot(221)
ax1.imshow(final_volume[:, :, mid_z], cmap='viridis')
ax1.set_title(f'Middle Z-slice (z={mid_z})')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Show middle Y-slice
mid_y = resolution // 2
ax2 = fig.add_subplot(222)
ax2.imshow(final_volume[:, mid_y, :], cmap='viridis')
ax2.set_title(f'Middle Y-slice (y={mid_y})')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

# Show middle X-slice
mid_x = resolution // 2
ax3 = fig.add_subplot(223)
ax3.imshow(final_volume[mid_x, :, :], cmap='viridis')
ax3.set_title(f'Middle X-slice (x={mid_x})')
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')

# Show 3D scatter plot of coral points
ax4 = fig.add_subplot(224, projection='3d')
coral_points = np.where(final_volume)
# Sample points for visualization (every 15th point to avoid overcrowding)
sample_indices = slice(0, len(coral_points[0]), 15)
ax4.scatter(coral_points[0][sample_indices], 
           coral_points[1][sample_indices], 
           coral_points[2][sample_indices], 
           c='coral', alpha=0.6, s=1)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('3D Coral Structure')

plt.tight_layout()
plt.savefig('_coral_v2_visualization.png', dpi=150, bbox_inches='tight')
print("Saved _coral_v2_visualization.png")
plt.show()
