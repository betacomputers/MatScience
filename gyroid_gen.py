import copy
import os
import trimesh

import numpy as np

from skimage import measure

# Lazy import of pyvista to avoid Python 3.14 compatibility issues
_pv = None
def get_pyvista():
    """Lazy import of pyvista to handle compatibility issues."""
    global _pv
    if _pv is None:
        try:
            # Workaround for Python 3.14 compatibility issue
            # The issue is in pyvista's _typing_core module trying to set __doc__ on Union
            import sys
            if sys.version_info >= (3, 14):
                # Try to patch the issue before importing
                import typing
                if hasattr(typing, 'Union'):
                    # This is a workaround - we can't fix it here, but we can provide better error
                    pass
            import pyvista as pv
            _pv = pv
        except (AttributeError, ImportError, TypeError) as e:
            # Handle Python 3.14 compatibility issue with pyvista
            error_msg = str(e)
            if "'typing.Union' object attribute '__doc__' is read-only" in error_msg or "read-only" in error_msg.lower():
                raise ImportError(
                    "PyVista is incompatible with Python 3.14 due to a typing issue.\n"
                    "Solutions:\n"
                    "  1. Use Python 3.11 or 3.12: conda create -n fenicsx-env python=3.11\n"
                    "  2. Update pyvista: pip install --upgrade pyvista (if a fix is available)\n"
                    "  3. Wait for pyvista to release a Python 3.14 compatible version"
                ) from e
            else:
                raise ImportError(
                    f"PyVista import failed: {e}\n"
                    "Try: pip install --upgrade pyvista"
                ) from e
    return _pv

# MAIN FUNCTIONS
# Plot TPMS equation:
def fn_plot_tpms_eq(tpms_type, tpms_design, sizes, cell_sizes, origin, unit_cell_mesh_resolution, c, thickness, mesh, show_plot=True):
    # Generation of the meshgrid:
    tols = [0, 0, 0]
    X, Y, Z, tols, spacing  = generate_meshgrid(0, sizes, cell_sizes, unit_cell_mesh_resolution)

    # Generate TPMS:
    F, t = tpms_library(X, Y, Z, c, tpms_design, cell_sizes, origin)

    # Mesh TPMS:
    if tpms_type == 'Shell':
        mesh, vertices = mesh_shell(F, t, thickness, sizes, mesh, tols, spacing)
    else:
        mesh, vertices = mesh_skeletal(F, sizes, mesh, tols, spacing)

    # Plot TPMS vertices (optional):
    if show_plot:
        try:
            # Colour TPMS vertices:
            color = []
            for vert in vertices:
                color.append(vert[0] * vert[1] * vert[2])
            color = np.array(color)

            pv = get_pyvista()
            plotter1 = pv.Plotter(window_size = [1400, 1600])
            _ = plotter1.add_mesh(vertices, scalars = color, cmap = 'jet')
            _ = plotter1.remove_scalar_bar()
            _ = plotter1.show_grid()
            plotter1.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
            print("Continuing without visualization...")

    return mesh, vertices

# Check face normals:
def fn_check_face_normals(mesh, silent = False, show_plot = True):
    # Calculate face centroids and normals:
    pv = get_pyvista()
    mesh_pv = pv.wrap(mesh)
    cent = mesh_pv.cell_centers().points
    direction = mesh_pv.cell_normals

    # Update output message:
    if not silent:
        print('\nCheck if face normals are pointing OUT of the mesh')

    # Plot TPMS face normals (optional):
    if show_plot:
        try:
            pv = get_pyvista()
            plotter2 = pv.Plotter(window_size = [1400, 1600])
            _ = plotter2.add_mesh(mesh, color = True, show_edges = True)
            _ = plotter2.add_arrows(cent, direction, mag = 1)
            _ = plotter2.remove_scalar_bar()
            _ = plotter2.show_grid()
            plotter2.show()
        except Exception as e:
            print(f"Warning: Could not display face normals plot: {e}")
            print("Continuing without visualization...")

    return mesh

# Check face normals:
def fn_flip_face_normals(mesh, silent = False, show_plot = True):
    # Flip mesh:
    pv = get_pyvista()
    mesh = pv.wrap(mesh)
    mesh.flip_faces(inplace=True)
    mesh = mesh_conversion(mesh)

    if not silent:
        print('Face normals were flipped. Now face normals should be pointing OUT of the mesh')

    # Plot TPMS face normals (optional):
    if show_plot:
        try:
            # Calculate face centroids and normals:
            pv = get_pyvista()
            mesh_pv = pv.wrap(mesh)
            cent = mesh_pv.cell_centers().points
            direction = mesh_pv.cell_normals
            
            pv = get_pyvista()
            plotter3 = pv.Plotter(window_size = [1400, 1600])
            _ = plotter3.add_mesh(mesh, color = True, show_edges = True)
            _ = plotter3.add_arrows(cent, direction, mag = 1)
            _ = plotter3.remove_scalar_bar()
            _ = plotter3.show_grid()
            plotter3.show()
        except Exception as e:
            print(f"Warning: Could not display face normals plot: {e}")
            print("Continuing without visualization...")

    return mesh

# Generate mesh:
def fn_generate_mesh(tpms_type, tpms_design, c, thickness, sizes, cell_sizes, origin, unit_cell_mesh_resolution, mesh, flip_face_normals, silent = False):
    is_watertight = False
    k = int(5 / 100 * unit_cell_mesh_resolution)
    k_max = int(45 / 100 * unit_cell_mesh_resolution)
    k_increment = int(5 / 100 * unit_cell_mesh_resolution)
    iterative_mesh = copy.deepcopy(mesh)

    # Mesh generation iterative process:
    if tpms_type == 'Shell':
        # Generate bounding box:
        shell_bounding_box = trimesh.creation.box(extents = (sizes[0], sizes[1], sizes[2]), transform = None)
        while not is_watertight and k <= k_max:
            # Generation of the meshgrid:
            X, Y, Z, tols, spacing  = generate_meshgrid(k, sizes, cell_sizes, unit_cell_mesh_resolution)
            
            # Generate TPMS for intersection:
            F, t = tpms_library(X, Y, Z, c, tpms_design, cell_sizes, origin)

            # Mesh TPMS for intersection:
            iterative_mesh, _ = mesh_shell(F, t, thickness, sizes, iterative_mesh, tols, spacing)

            # Check face normals orientation:
            if flip_face_normals:
                pv = get_pyvista()
                iterative_mesh = pv.wrap(iterative_mesh)
                iterative_mesh.flip_faces(inplace=True)
                iterative_mesh = mesh_conversion(iterative_mesh)
            
            # Ensure mesh is valid for boolean operations:
            try:
                # Try to repair the mesh first
                iterative_mesh.remove_duplicate_faces()
                iterative_mesh.remove_unreferenced_vertices()
                # For shell meshes, try to make them watertight
                if not iterative_mesh.is_watertight:
                    try:
                        iterative_mesh.fill_holes()
                    except:
                        pass
            except Exception as e:
                if not silent:
                    print(f"Could not repair mesh at k={k}: {e}")
            
            # For shell meshes, clip to bounding box using spatial filtering instead of boolean ops
            # This works better for surface meshes
            try:
                # Convert to PyVista for clipping
                pv = get_pyvista()
                mesh_pv = pv.wrap(iterative_mesh)
                
                # Create bounding box as PyVista object
                bounds = [
                    -sizes[0]/2, sizes[0]/2,
                    -sizes[1]/2, sizes[1]/2,
                    -sizes[2]/2, sizes[2]/2
                ]
                
                # Clip mesh to bounding box
                clipped_mesh = mesh_pv.clip_box(bounds, invert=False)
                
                # Extract surface to ensure we have triangular faces
                if clipped_mesh.n_cells > 0:
                    try:
                        # Try to extract surface which gives us triangular faces
                        surface_mesh = clipped_mesh.extract_surface()
                        if surface_mesh.n_cells > 0:
                            iterative_mesh = mesh_conversion(surface_mesh)
                        else:
                            # Fallback: try direct conversion
                            iterative_mesh = mesh_conversion(clipped_mesh)
                    except:
                        # Fallback: try direct conversion
                        iterative_mesh = mesh_conversion(clipped_mesh)
                else:
                    if not silent:
                        print(f"Warning: Clipped mesh is empty at k={k}")
                    k += k_increment
                    continue
                    
            except Exception as e:
                if not silent:
                    print(f"Mesh clipping failed at k={k}: {e}, trying boolean operation...")
                # Fallback to boolean operation if clipping fails
                try:
                    # Try to make mesh watertight first
                    if not iterative_mesh.is_watertight:
                        iterative_mesh.fill_holes()
                    
                    # Only try boolean if mesh is watertight
                    if iterative_mesh.is_watertight:
                        iterative_mesh = trimesh.boolean.intersection((iterative_mesh, shell_bounding_box), engine = 'blender')
                    else:
                        if not silent:
                            print(f"Mesh is not watertight, skipping boolean operation at k={k}")
                        k += k_increment
                        continue
                except Exception as e2:
                    if not silent:
                        print(f"Boolean intersection also failed at k={k}: {e2}")
                    k += k_increment
                    continue
            
            # Check obtained results
            k += k_increment
            is_watertight = iterative_mesh.is_watertight
            if not iterative_mesh.is_watertight:
                iterative_mesh.fill_holes()
                is_watertight = iterative_mesh.is_watertight
    else:
        # Generate bounding box:
        bounding_box_1 = trimesh.creation.box(extents = (2 * sizes[0], 2 * sizes[1], 2 * sizes[2]), transform = None)
        bounding_box_2 = trimesh.creation.box(extents = (sizes[0], sizes[1], sizes[2]), transform = None)
        bounding_box = trimesh.boolean.difference((bounding_box_1, bounding_box_2), engine = 'blender')
        
        del bounding_box_1, bounding_box_2
        
        while not is_watertight and k <= k_max:
            # Generation of the meshgrid:
            X, Y, Z, tols, spacing  = generate_meshgrid(k, sizes, cell_sizes, unit_cell_mesh_resolution)
            
            # Generate TPMS for intersection:
            F, t = tpms_library(X, Y, Z, c, tpms_design, cell_sizes, origin)

            # Mesh TPMS for intersection:
            iterative_mesh, _ = mesh_skeletal(F, sizes, iterative_mesh, tols, spacing)

            # Check face normals orientation:
            if flip_face_normals:
                pv = get_pyvista()
                iterative_mesh = pv.wrap(iterative_mesh)
                iterative_mesh.flip_faces(inplace=True)
                iterative_mesh = mesh_conversion(iterative_mesh)
            
            # Ensure mesh is valid for boolean operations:
            try:
                # Try to repair the mesh first
                iterative_mesh.remove_duplicate_faces()
                iterative_mesh.remove_unreferenced_vertices()
                # For skeletal meshes, try to make them watertight
                if not iterative_mesh.is_watertight:
                    try:
                        iterative_mesh.fill_holes()
                    except:
                        pass
            except Exception as e:
                if not silent:
                    print(f"Could not repair mesh at k={k}: {e}")
            
            # For skeletal meshes, use PyVista clipping to create the hollow structure
            # This avoids boolean operations which require volumes
            try:
                # Convert to PyVista
                pv = get_pyvista()
                mesh_pv = pv.wrap(iterative_mesh)
                
                # Create outer and inner bounding boxes
                outer_bounds = [
                    -sizes[0], sizes[0],
                    -sizes[1], sizes[1],
                    -sizes[2], sizes[2]
                ]
                inner_bounds = [
                    -sizes[0]/2, sizes[0]/2,
                    -sizes[1]/2, sizes[1]/2,
                    -sizes[2]/2, sizes[2]/2
                ]
                
                # Clip to outer box first
                clipped_outer = mesh_pv.clip_box(outer_bounds, invert=False)
                
                # Then clip to remove inner box (invert=True removes what's inside)
                if clipped_outer.n_cells > 0:
                    clipped_final = clipped_outer.clip_box(inner_bounds, invert=True)
                    
                    # Extract surface and convert back to trimesh
                    if clipped_final.n_cells > 0:
                        try:
                            # Try to extract surface which gives us triangular faces
                            surface_mesh = clipped_final.extract_surface()
                            if surface_mesh.n_cells > 0:
                                iterative_mesh = mesh_conversion(surface_mesh)
                            else:
                                # Fallback: try direct conversion
                                iterative_mesh = mesh_conversion(clipped_final)
                        except:
                            # Fallback: try direct conversion
                            iterative_mesh = mesh_conversion(clipped_final)
                    else:
                        if not silent:
                            print(f"Warning: Clipped skeletal mesh is empty at k={k}")
                        k += k_increment
                        continue
                else:
                    if not silent:
                        print(f"Warning: Outer clipped mesh is empty at k={k}")
                    k += k_increment
                    continue
                    
            except Exception as e:
                if not silent:
                    print(f"Mesh clipping failed at k={k}: {e}, trying boolean operation...")
                # Fallback to boolean operation if clipping fails
                try:
                    # Try to make mesh watertight first
                    if not iterative_mesh.is_watertight:
                        iterative_mesh.fill_holes()
                    
                    # Only try boolean if mesh is watertight
                    if iterative_mesh.is_watertight:
                        iterative_mesh = trimesh.boolean.difference((iterative_mesh, bounding_box), engine = 'blender')
                    else:
                        if not silent:
                            print(f"Mesh is not watertight, skipping boolean operation at k={k}")
                        k += k_increment
                        continue
                except Exception as e2:
                    if not silent:
                        print(f"Boolean difference also failed at k={k}: {e2}")
                    k += k_increment
                    continue
            
            # Check obtained results
            k += k_increment
            is_watertight = iterative_mesh.is_watertight
            if not iterative_mesh.is_watertight:
                iterative_mesh.fill_holes()
                is_watertight = iterative_mesh.is_watertight

    # Update output message:
    if not silent:
        if is_watertight:
            print('Mesh is generated!')
            print('The obtained mesh is watertight. If the opposite solution was desired, try using the opposite face normals direction.')
        else:
            print('Mesh is generated:')
            print('Cannot obtain a watertight mesh. Try increasing unit cell mesh resolution. Please, check results carefully and treat them to solve this issue.')
    
    # Plot generated mesh (optional):
    if not silent:
        try:
            pv = get_pyvista()
            plotter4 = pv.Plotter(window_size = [1400, 1600])
            _ = plotter4.add_title('Generated mesh can be exported into STL format', font_size = 10)
            _ = plotter4.add_mesh(iterative_mesh, color = True, show_edges = True)
            _ = plotter4.show_grid()
            plotter4.show()
        except Exception as e:
            print(f"Warning: Could not display final mesh plot: {e}")
            print("Mesh generation completed successfully, but visualization was skipped.")

    return iterative_mesh

# Export mesh:
def fn_export_stl_file(iterative_mesh, file_name, directory_path, silent = False):
    export = trimesh.exchange.stl.export_stl_ascii(iterative_mesh)
    
    file_path = os.path.join(directory_path, file_name + '.stl')
    with open(file_path, 'w') as file:
        file.write(export)
    
    if not silent:
        print('\nMesh exported as .STL into ' + file_path)

# SUPLEMENTARY FUNCTIONS:
# Generate meshgrid
def generate_meshgrid(k, sizes, cell_sizes, unit_cell_mesh_resolution):
    tol_x = k * cell_sizes[0] / unit_cell_mesh_resolution
    tol_y = k * cell_sizes[1] / unit_cell_mesh_resolution
    tol_z = k * cell_sizes[2] / unit_cell_mesh_resolution
    tols = [tol_x, tol_y, tol_z]

    xl = np.linspace(-sizes[0]/2 - tols[0], sizes[0]/2 + tols[0], int(sizes[0] / cell_sizes[0]) * unit_cell_mesh_resolution + 2 * k + 1)
    yl = np.linspace(-sizes[1]/2 - tols[1], sizes[1]/2 + tols[1], int(sizes[1] / cell_sizes[1]) * unit_cell_mesh_resolution + 2 * k + 1)
    zl = np.linspace(-sizes[2]/2 - tols[2], sizes[2]/2 + tols[2], int(sizes[2] / cell_sizes[2]) * unit_cell_mesh_resolution + 2 * k + 1)
    spacing = [xl, yl, zl]
    
    Y, X, Z = np.meshgrid(yl, xl, zl)

    return X, Y, Z, tols, spacing  

# Mesh conversion
def mesh_conversion(mesh_pv):
    # PyVista meshes store faces in VTK format: [n, v1, v2, v3, n, v1, v2, v3, ...]
    # where n is the number of vertices in the face
    
    try:
        # Get points
        points = np.array(mesh_pv.points)
        
        # Get faces - PyVista uses VTK format
        if hasattr(mesh_pv, 'faces') and mesh_pv.faces is not None:
            faces_flat = np.array(mesh_pv.faces)
        elif hasattr(mesh_pv, 'cell_faces'):
            # Alternative: get faces from cells
            faces_flat = np.array(mesh_pv.cell_faces)
        else:
            # Try to extract surface
            try:
                surface = mesh_pv.extract_surface()
                if surface.faces is not None:
                    faces_flat = np.array(surface.faces)
                else:
                    raise ValueError("No faces found in mesh")
            except:
                raise ValueError("No faces found in mesh")
        
        # Parse VTK face format
        if len(faces_flat) == 0:
            raise ValueError("Empty faces array")
        
        faces_list = []
        i = 0
        while i < len(faces_flat):
            n_verts = int(faces_flat[i])
            if n_verts == 3:
                # Triangle face
                faces_list.append([int(faces_flat[i+1]), int(faces_flat[i+2]), int(faces_flat[i+3])])
                i += 4
            elif n_verts > 3:
                # Polygon - triangulate it (simple fan triangulation)
                base_idx = int(faces_flat[i+1])
                for j in range(2, n_verts):
                    faces_list.append([
                        base_idx,
                        int(faces_flat[i+j]),
                        int(faces_flat[i+j+1])
                    ])
                i += n_verts + 1
            else:
                i += n_verts + 1
        
        if len(faces_list) == 0:
            raise ValueError("No valid faces found after parsing")
        
        faces_as_array = np.array(faces_list)
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=points, faces=faces_as_array, process=False)
        return mesh
        
    except Exception as e:
        # Last resort: try using trimesh's built-in conversion if available
        try:
            # Try to save and reload
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                mesh_pv.save(tmp.name)
                mesh = trimesh.load(tmp.name)
                os.unlink(tmp.name)
                return mesh
        except:
            raise ValueError(f"Mesh conversion failed: {e}")

# Mesh Shell
def mesh_shell(F, t, thickness, sizes, mesh, tols, spacing):
    vertices_positive, faces_positive, vertex_normals_positive, _ = measure.marching_cubes(F, thickness * t, spacing = [np.diff(spacing[0])[0], np.diff(spacing[1])[0], np.diff(spacing[2])[0]])
    vertices_negative, faces_negative, vertex_normals_negative, _ = measure.marching_cubes(F, -thickness * t, spacing = [np.diff(spacing[0])[0], np.diff(spacing[1])[0], np.diff(spacing[2])[0]])

    for i, vert in enumerate(vertices_positive):
        vertices_positive[i, 0] = vert[0] - sizes[0]/2 - tols[0]
        vertices_positive[i, 1] = vert[1] - sizes[1]/2 - tols[1]
        vertices_positive[i, 2] = vert[2] - sizes[2]/2 - tols[2]

    for i, vert in enumerate(vertices_negative):
        vertices_negative[i, 0] = vert[0] - sizes[0]/2 - tols[0]
        vertices_negative[i, 1] = vert[1] - sizes[1]/2 - tols[1]
        vertices_negative[i, 2] = vert[2] - sizes[2]/2 - tols[2]

    vertices = np.concatenate((vertices_positive, vertices_negative))
    
    # Create meshes without vertex_normals to avoid computation issues with degenerate triangles
    try:
        mesh_1 = trimesh.Trimesh(vertices = vertices_positive, faces = faces_positive, process=False)
        # Clean up mesh_1 before processing
        mesh_1.remove_duplicate_faces()
        mesh_1.remove_unreferenced_vertices()
        # Remove degenerate faces (zero area triangles)
        if len(mesh_1.faces) > 0:
            areas = mesh_1.area_faces
            valid_faces = areas > 1e-10  # Remove faces with very small area
            if np.any(~valid_faces):
                mesh_1.update_faces(valid_faces)
    except Exception as e:
        # If mesh creation fails, return empty mesh
        return trimesh.Trimesh(), vertices
    
    try:
        mesh_2 = trimesh.Trimesh(vertices = vertices_negative, faces = faces_negative, process=False)
        # Clean up mesh_2 before processing
        mesh_2.remove_duplicate_faces()
        mesh_2.remove_unreferenced_vertices()
        # Remove degenerate faces
        if len(mesh_2.faces) > 0:
            areas = mesh_2.area_faces
            valid_faces = areas > 1e-10
            if np.any(~valid_faces):
                mesh_2.update_faces(valid_faces)
        
        # Flip faces using PyVista
        pv = get_pyvista()
        mesh_2_pv = pv.wrap(mesh_2)
        mesh_2_pv.flip_faces(inplace=True)
        mesh_2 = mesh_conversion(mesh_2_pv)
    except Exception as e:
        # If mesh_2 creation fails, just use mesh_1
        mesh = mesh_1
        del vertices_positive, faces_positive, vertex_normals_positive, vertices_negative, faces_negative, vertex_normals_negative, mesh_1
        return mesh, vertices

    # Concatenate meshes
    try:
        mesh = trimesh.util.concatenate((mesh_1, mesh_2))
    except Exception as e:
        # If concatenation fails, try to process meshes first
        try:
            mesh_1.process()
            mesh_2.process()
            mesh = trimesh.util.concatenate((mesh_1, mesh_2))
        except:
            # Last resort: just use mesh_1
            mesh = mesh_1
    
    # Try to merge vertices and clean up the mesh
    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        # Remove any remaining degenerate faces
        if len(mesh.faces) > 0:
            areas = mesh.area_faces
            valid_faces = areas > 1e-10
            if np.any(~valid_faces):
                mesh.update_faces(valid_faces)
    except:
        pass
    
    del vertices_positive, faces_positive, vertex_normals_positive, vertices_negative, faces_negative, vertex_normals_negative, mesh_1, mesh_2

    return mesh, vertices

# Mesh Skeletal
def mesh_skeletal(F, sizes, mesh, tols, spacing):
    vertices, faces, _, _ = measure.marching_cubes(F, 0, spacing = [np.diff(spacing[0])[0], np.diff(spacing[1])[0], np.diff(spacing[2])[0]])
    for i, vert in enumerate(vertices):
        vertices[i, 0] = vert[0] - sizes[0]/2 - tols[0]
        vertices[i, 1] = vert[1] - sizes[1]/2 - tols[1]
        vertices[i, 2] = vert[2] - sizes[2]/2 - tols[2]
    
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)

    del faces

    return mesh, vertices

# TPMS library
def tpms_library(X, Y, Z, c, tpms_design, cell_sizes, origin, silent = False):
    w_x = 1 / cell_sizes[0] * 2 * np.pi
    w_y = 1 / cell_sizes[1] * 2 * np.pi
    w_z = 1 / cell_sizes[2] * 2 * np.pi
    
    if tpms_design == 'Skeletal-TPMS Schoen gyroid' or tpms_design == 'Shell-TPMS Gyroid':
        F = (np.cos(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) + np.cos(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) + np.cos(w_z * (Z + origin[2])) * np.sin(w_x * (X + origin[0])) - c)   # J
        t = 0.125
    
    elif tpms_design == 'Skeletal-TPMS Schwarz diamond':
        F = (np.cos(w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.sin(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) - c)   # K
        t = 0
    
    elif tpms_design == 'Skeletal-TPMS Schwarz primitive (pinched)' or tpms_design == 'Skeletal-TPMS Schwarz primitive':
        F = (np.cos(w_x * (X + origin[0])) + np.cos(w_y * (Y + origin[1])) + np.cos(w_z * (Z + origin[2])) - c)   # M, N
        t = 0
    
    elif tpms_design == 'Skeletal-TPMS Body diagonals with nodes':
        F = (2 * (np.cos(w_x * ((X + origin[0]))) * np.cos(w_y * (Y + origin[1])) + np.cos(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_z * (Z + origin[2])) * np.cos(w_x * (X + origin[0]))) - (np.cos(2 * w_x * (X + origin[0])) + np.cos(2 * w_y * (Y + origin[1])) + np.cos(2 * w_z * (Z + origin[2]))) - c) # O
        t = 0
    
    elif tpms_design == 'Shell-TPMS Diamond':
        F = (np.sin(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) + np.sin(w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) - c) # Q
        t = 0.115
    
    elif tpms_design == 'Shell-TPMS Lidinoid':
        F = (np.sin(2 * w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) + np.sin(w_x * (X + origin[0])) * np.sin(2 * w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.sin(2 * w_z * (Z + origin[2])) - np.cos(2 * w_x * (X + origin[0])) * np.cos(2 * w_y * (Y + origin[1])) - np.cos(2 * w_y * (Y + origin[1])) * np.cos(2 * w_z * (Z + origin[2])) - np.cos(2 * w_z * (Z + origin[2])) * np.cos(2 * w_x * (X + origin[0])) + 0.3 - c)
        t = 0.37
    
    elif tpms_design == 'Shell-TPMS Split-P':
        F = (1.1 * (np.sin(2 * w_x * (X + origin[0])) * np.cos(w_y * (Y + origin[1])) * np.sin(w_z * (Z + origin[2])) + np.sin(w_x * (X + origin[0])) * np.sin(2 * w_y * (Y + origin[1])) * np.cos(w_z * (Z + origin[2])) + np.cos(w_x * (X + origin[0])) * np.sin(w_y * (Y + origin[1])) * np.sin(2 * w_z * (Z + origin[2]))) - 0.2 * (np.cos(2 * w_x * (X + origin[0])) * np.cos(2 * w_y * (Y + origin[1])) + np.cos(2 * w_y * (Y + origin[1])) * np.cos(2 * w_z * (Z + origin[2])) + np.cos(2 * w_z * (Z + origin[2])) * np.cos(2 * w_x * (X + origin[0]))) - 0.4 * (np.cos(2 * w_x * (X + origin[0])) + np.cos(2 * w_y * (Y + origin[1])) + np.cos(2 * w_z * (Z + origin[2]))) - c)
        t = 0.19
    
    elif tpms_design == 'Shell-TPMS Schwarz':
        F = (np.cos(w_x * (X + origin[0])) + np.cos(w_y * (Y + origin[1])) + np.cos(w_z * (Z + origin[2])) - c)
        t = 0.0875
    
    else:
        if not silent:
            print('Design not found in library')
        F = 0
        t = 0

    return F, t