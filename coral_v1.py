import numpy as np
from skimage import measure
from stl import mesh
import matplotlib.pyplot as plt

# Parameters
container_size = 20.0  # mm - size of rectangular container (20)
resolution = 80  # points per dimension (80)
max_iterations = 5  # fractal iteration depth (4)
branch_probability = 1.0  # probability of branching (0.7)
branch_angle = np.pi/4  # angle between branches (45 degrees)
branch_length_factor = 0.6  # how much shorter each branch is (0.6)
branch_thickness_factor = 0.7  # how much thinner each branch is (0.7)
base_thickness = 1.2  # mm - thickness of main branches (1.2)
num_initial_branches = 8  # number of main branches from center (8)

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
    curve_magnitude = 0.15
    curve_direction = np.array([np.random.uniform(-1, 1) for _ in range(3)])
    curve_direction = curve_direction / np.linalg.norm(curve_direction)
    
    # Create curved path for the branch
    num_segments = max(3, int(length * 4))
    segment_length = length / num_segments
    
    current_pos = start_point
    current_direction = direction
    
    for i in range(num_segments):
        # Calculate next position with curve
        next_pos = current_pos + current_direction * segment_length
        
        # Add curve to direction
        curve_effect = curve_direction * curve_magnitude * segment_length
        next_pos += curve_effect
        
        # Create cylinder segment
        create_cylinder_segment(current_pos, next_pos, thickness)
        
        # Update for next segment - check for zero length
        direction_vector = next_pos - current_pos
        direction_length = np.linalg.norm(direction_vector)
        
        if direction_length > 0.01:  # Only update if there's meaningful movement
            current_direction = direction_vector / direction_length
        
        current_pos = next_pos
    
    # Decide whether to create sub-branches
    if iteration < max_iterations - 1 and np.random.random() < branch_probability:
        # Create 1-3 sub-branches
        num_sub_branches = np.random.randint(1, 4)
        
        for i in range(num_sub_branches):
            # Calculate new direction with some randomness
            angle = branch_angle * (1 + np.random.uniform(-0.3, 0.3))
            
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
                
                # Create sub-branch
                new_length = length * branch_length_factor * (0.6 + 0.8 * np.random.random())
                new_thickness = thickness * branch_thickness_factor
                new_start = current_pos  # Start from current position
                
                create_branch(new_start, new_direction, new_length, new_thickness, iteration + 1)

def generate_coral():
    """Generate a coral structure with multiple main branches"""
    center = np.array([0, 0, 0])
    
    # Create multiple main branches growing outward from center
    for i in range(num_initial_branches):
        # Create direction in spherical coordinates for even distribution
        phi = np.pi * (1 - np.sqrt(5)) / 2  # golden angle
        theta = 2 * np.pi * i / num_initial_branches
        
        # Convert to Cartesian coordinates
        x_dir = np.sin(phi) * np.cos(theta)
        y_dir = np.sin(phi) * np.sin(theta)
        z_dir = np.cos(phi)
        
        # Add some randomness to make it more natural
        direction = np.array([x_dir, y_dir, z_dir])
        direction += np.random.uniform(-0.3, 0.3, 3)
        direction = direction / np.linalg.norm(direction)
        
        # Randomize branch length and thickness slightly
        branch_length = container_size * 0.4 * (0.7 + 0.6 * np.random.random())
        branch_thickness = base_thickness * (0.8 + 0.4 * np.random.random())
        
        print(f"Creating main branch {i+1}/{num_initial_branches}")
        create_branch(center, direction, branch_length, branch_thickness, 0)

# Generate the coral structure
print("Generating coral structure...")
generate_coral()

# Apply container bounds (rectangular container)
container_mask = (X >= -container_size/2) & (X <= container_size/2) & \
                 (Y >= -container_size/2) & (Y <= container_size/2) & \
                 (Z >= -container_size/2) & (Z <= container_size/2)

final_volume = coral_volume & container_mask

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

# Save STL file
coral_mesh.save('_coral.stl')
print(f"Saved _coral.stl with {len(faces)} faces")

# Optional: Create a simple visualization
print("Creating visualization...")
fig = plt.figure(figsize=(12, 8))

# Show middle slice
"""mid_z = resolution // 2
ax1 = fig.add_subplot(221)
ax1.imshow(final_volume[:, :, mid_z], cmap='viridis')
ax1.set_title(f'Middle Z-slice (z={mid_z})')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')"""

# Show middle Y-slice
"""mid_y = resolution // 2
ax2 = fig.add_subplot(222)
ax2.imshow(final_volume[:, mid_y, :], cmap='viridis')
ax2.set_title(f'Middle Y-slice (y={mid_y})')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')"""

# Show middle X-slice
"""mid_x = resolution // 2
ax3 = fig.add_subplot(223)
ax3.imshow(final_volume[mid_x, :, :], cmap='viridis')
ax3.set_title(f'Middle X-slice (x={mid_x})')
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')"""

# Show 3D scatter plot of coral points
ax4 = fig.add_subplot(111, projection='3d')
coral_points = np.where(final_volume)
# Sample points for visualization (every 15th point to avoid overcrowding)
sample_indices = slice(0, len(coral_points[0]), 15)
# Map voxel indices to physical coordinates
px = x[coral_points[0][sample_indices]]
py = y[coral_points[1][sample_indices]]
pz = z[coral_points[2][sample_indices]]
ax4.scatter(px, py, pz, c='coral', alpha=0.6, s=1)
ax4.set_xlabel('X (mm)')
ax4.set_ylabel('Y (mm)')
ax4.set_zlabel('Z (mm)')
ax4.set_title('3D Coral Structure')

# Center the 3D visualization
ax4.set_box_aspect([1, 1, 1])  # Equal aspect ratio
ax4.set_xlim(x.min(), x.max())
ax4.set_ylim(y.min(), y.max())
ax4.set_zlim(z.min(), z.max())

plt.tight_layout()
plt.savefig('_coral_visualization.png', dpi=150, bbox_inches='tight')
print("Saved _coral_visualization.png")
plt.show()
