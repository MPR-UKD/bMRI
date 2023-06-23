import numpy as np
from pathlib import Path
from tvtk.api import tvtk
from mayavi import mlab
from src.Utilitis import load_nii, get_dcm_list, get_dcm_array
from src.Fitting.T2_T2star import T2_T2star
from src.Fitting.T1rho_T2prep import T1rho_T2prep


def load_image(file_path: Path) -> np.ndarray:
    """
    Load the image and apply the mask if available.

    :param file_path: Path to the .nii image file
    :return: Image array with mask applied
    """
    image = load_nii(file_path).array
    if 'value' in file_path.name:
        mask = load_nii(file_path.parent / 'mask.nii.gz').array
        image[mask == 0] = 0
    return image


def create_vtk_data(non_zero_indices: tuple, values: np.ndarray) -> tvtk.PolyData:
    """
    Convert the data into a VTK object.

    :param non_zero_indices: Non-zero indices of the image
    :param values: Non-zero values of the image
    :return: VTK PolyData object
    """
    pts = np.vstack(non_zero_indices).T
    vtk_data = tvtk.PolyData(points=pts)
    vtk_data.point_data.scalars = values
    vtk_data.point_data.scalars.name = 'Intensity'
    return vtk_data


def add_colorbar(nodes: mlab.GlyphSource, font_size: int) -> None:
    """
    Add a color bar to the visualization.

    :param nodes: Nodes to which color bar is attached
    :param font_size: Font size for the color bar label
    """
    colorbar = mlab.colorbar(nodes, title=" ms", orientation='vertical', label_fmt='%.0f')
    colorbar.scalar_bar.unconstrained_font_size = True
    colorbar.label_text_property.font_size = font_size
    colorbar.title_text_property.font_size = round(font_size * 2)


def save_img(file_path: Path, filename: str, dicom_processor: callable, spacing: list, min_val: float,
             max_val: float) -> None:
    """
    Load and visualize an image, and save the visualization to a file.

    :param file_path: Path to the .nii image file
    :param filename: Output file name
    :param dicom_processor: Processor for DICOM data
    :param spacing: Spacing between nodes
    :param min_val: Minimum value for color scale
    :param max_val: Maximum value for color scale
    """
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))

    # Load image
    image = load_image(file_path)

    # Get non-zero indices and values
    non_zero_indices = np.nonzero(image)
    values = image[non_zero_indices]

    # Scale values for color mapping
    scaled_values = (values - min_val) / (max_val - min_val)

    # Visualize nodes with color mapping
    nodes = mlab.points3d(
        non_zero_indices[0] * spacing[0],
        non_zero_indices[1] * spacing[1],
        non_zero_indices[2] * spacing[2],
        scale_factor=2,
        mode='cube',
        colormap='jet',
        opacity=1
    )
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = scaled_values

    # Convert data to VTK object
    vtk_data = create_vtk_data(non_zero_indices, values)

    # Add dataset to visualization pipeline and create color bar
    src = mlab.pipeline.add_dataset(vtk_data)
    nodes2 = mlab.pipeline.glyph(src, scale_factor=2, mode='cube', colormap='jet', opacity=0)
    nodes2.glyph.scale_mode = 'scale_by_vector'
    nodes2.module_manager.scalar_lut_manager.data_range = [min_val, max_val]

    # Load DICOM data and adjust font size
    if dicom_processor:
        data, _ = dicom_processor.read_data(Path(file_path).parent)
        data = data[0][:, :, ::-1]
        font_size = 25
    else:
        data = get_dcm_array(get_dcm_list([_ for _ in file_path.parent.parent.glob('*dGEMRIC*')][0])).transpose(
            (2, 1, 0))[:, :, ::-1]
        font_size = 30

    # Add color bar to visualization
    add_colorbar(nodes2, font_size)

    mask = image
    mask[mask != 0] = 1
    mask_coordinates = np.where(mask)

    src = mlab.pipeline.scalar_field(data)
    src.spacing = spacing
    src.update_image_data = True

    thr = mlab.pipeline.threshold(src, low=0)
    rescaled_coordinates = (mask_coordinates[0] * spacing[0],
                            mask_coordinates[1] * spacing[1],
                            mask_coordinates[2] * spacing[2])
    avg_coordinates = tuple(np.median(coordinate) for coordinate in rescaled_coordinates)

    vmin = np.percentile(data[data != 0], 5)
    vmax = np.percentile(data[data != 0], 95)
    print(filename, vmin, vmax)
    for plane_orientation, opacity in [('z_axes', 1), ('y_axes', 0.5)]:
        cut_plane = mlab.pipeline.scalar_cut_plane(thr,
                                                   plane_orientation=plane_orientation,
                                                   colormap='black-white',
                                                   vmin=vmin,
                                                   vmax=vmax,
                                                   opacity=opacity)
        cut_plane.implicit_plane.origin = avg_coordinates
        cut_plane.implicit_plane.widget.enabled = False

    mlab.view(azimuth=35.264389682754654, elevation=-45.0, distance=450)
    mlab.roll(180)
    mlab.savefig(filename + '_1.pdf')
    try:
        mlab.close(fig)
    except:
        pass


def process_image(pre_post, imgs, knee, file_pattern, output_suffix, dicom_processor, spacing, min_val, max_val):
    nii_file = [_ for _ in pre_post.glob(file_pattern)][0]
    save_img(nii_file, str(imgs / knee.name.split('tion_')[1].split('_')[0]) + output_suffix, dicom_processor,
             spacing, min_val, max_val)


if __name__ == '__main__':
    root = Path(r'E:\Buckup\Projekt_Schweineknie\Daten')
    imgs = Path(r'C:\Users\ludge\Downloads\Bild')
    # if imgs.exists():
    #    shutil.rmtree(imgs)
    #    os.mkdir(imgs)
    nr = 40
    for knee in root.glob('*'):
        pre = True
        try:
            if int(knee.name.split('tion_')[1].split('_')[0]) != nr:
                continue
            for pre_post in knee.glob('*'):
                if pre:
                    process_image(pre_post, imgs, knee, '*T1_Images*/value_map.nii.gz', "_T1", None,
                                  [1, 1, 2], min_val=600, max_val=1200)
                    process_image(pre_post, imgs, knee, '*T2_map*/t2_t2star_map.nii.gz', "_T2", T2_T2star(dim=3),
                                  [1, 1, 4],
                                  min_val=20, max_val=80)
                    process_image(pre_post, imgs, knee, 'T1rho/t1rho_map.nii.gz', "_T1rho", T1rho_T2prep(dim=3),
                                  [1, 1, 2],
                                  min_val=40, max_val=200)
                    process_image(pre_post, imgs, knee, '*T2-star*/t2_t2star_map.nii.gz', "_T2star", T2_T2star(dim=3),
                                  [1, 1, 2], min_val=0, max_val=50)
                else:
                    process_image(pre_post, imgs, knee, '*T1_Images*/value_map.nii.gz', "_dGEMRIC", None,
                                  [1, 1, 2], min_val=300, max_val=1000)
                pre = False
        except Exception as e:
            pass
