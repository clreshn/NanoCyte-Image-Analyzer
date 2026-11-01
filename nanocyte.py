"""
NanoCyte
===============================

Quantifies cell-associated nanoparticle fluorescence using two separate,
spatially aligned TIFF images:
  1. DAPI image: used for nucleus segmentation.
  2. Nanoparticle (NP) image: used for fluorescence quantification.

For each DAPI-segmented nucleus, a territory is expanded outward by
``cell_radius`` pixels. Where territories overlap, pixels are assigned to
the nearest nucleus. 

Outputs include cell-level measurements, segmentation/territory overlays,
a labeled nucleus-mask TIFF, and a background-subtracted fluorescence TIFF.

"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries

warnings.filterwarnings("ignore", category=RuntimeWarning)


class CellFluorescenceAnalyzer:
    """Measure nanoparticle fluorescence around DAPI-segmented nuclei."""

    def __init__(
        self,
        min_nucleus_area=150,
        max_nucleus_area=5000,
        min_circularity=0.25,
        cell_radius=20,
        background_sigma=10,
        dapi_background_sigma=30,
        threshold_factor=0.05,
        min_object_size=40,
        watershed_min_distance=5,
    ):
        """
        Parameters
        ----------
        min_nucleus_area, max_nucleus_area : int
            Accepted nucleus area range, in pixels.
        min_circularity : float
            Minimum object circularity (1 = perfect circle). Set to 0 to
            disable circularity filtering. A lower default is used here to
            be more permissive toward irregular nuclei.
        cell_radius : int
            Maximum distance in pixels that a measurement territory extends
            outward from the edge of a nucleus.
        background_sigma : float
            Gaussian sigma used to estimate slowly varying NP-image
            background. This is in pixels.
        dapi_background_sigma : float
            Gaussian sigma for estimating broad DAPI background before
            thresholding. This is in pixels.
        threshold_factor : float
            Multiplier applied to the Otsu threshold after DAPI background
            subtraction. Values below 1 are more permissive; lower values
            can also cause nearby nuclei to merge. Inspect the segmentation
            output when changing this parameter.
        min_object_size : int
            Smallest connected DAPI foreground object retained before
            watershed splitting.
        watershed_min_distance : int
            Minimum spacing, in pixels, between candidate watershed markers.
            Reduce this if touching nuclei are being merged; increase it if
            individual nuclei are being split into multiple objects.
        """
        if cell_radius < 0:
            raise ValueError("cell_radius must be >= 0.")
        if min_nucleus_area < 1:
            raise ValueError("min_nucleus_area must be >= 1.")
        if max_nucleus_area < min_nucleus_area:
            raise ValueError(
                "max_nucleus_area must be >= min_nucleus_area."
            )
        if not 0 <= min_circularity <= 1:
            raise ValueError("min_circularity must be between 0 and 1.")
        if threshold_factor <= 0:
            raise ValueError("threshold_factor must be > 0.")

        self.min_nucleus_area = int(min_nucleus_area)
        self.max_nucleus_area = int(max_nucleus_area)
        self.min_circularity = float(min_circularity)
        self.cell_radius = int(cell_radius)
        self.background_sigma = float(background_sigma)
        self.dapi_background_sigma = float(dapi_background_sigma)
        self.threshold_factor = float(threshold_factor)
        self.min_object_size = int(min_object_size)
        self.watershed_min_distance = int(watershed_min_distance)

        self.dapi_path = None
        self.np_path = None
        self.output_stem = None
        self.output_dir = None

        self.dapi = None
        self.np_image = None
        self.dapi_corrected = None
        self.dapi_threshold = None
        self.dapi_binary = None
        self.nuclei = None
        self.background = None
        self.corrected_fluorescence = None
        self.cell_territories = None
        self.results = None

    # Image input
    @staticmethod
    def _load_grayscale_tiff(image_path, image_role):
        """Load a grayscale or RGB TIFF and extract the appropriate channel."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"{image_role} image not found: {image_path}"
            )

        image = tiff.imread(image_path)

        if image.size == 0:
            raise ValueError(f"{image_role} image is empty: {image_path}")

        if not np.issubdtype(image.dtype, np.number):
            raise ValueError(f"{image_role} image must contain numeric pixels.")

        # Handle grayscale TIFFs with singleton-channel dimensions.
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        # For RGB/RGBA images: DAPI = blue channel; NP = red channel.
        elif image.ndim == 3 and image.shape[-1] in (3, 4):
            channel = 2 if image_role.lower() == "dapi" else 0
            image = image[..., channel]

        if image.ndim != 2:
            raise ValueError(
                f"Unexpected image shape for {image_role}: {image.shape}"
            )

        return image.astype(np.float32, copy=False)

    def load_images(self, dapi_path, np_path):
        """Load and validate the separate DAPI and NP images."""
        self.dapi_path = Path(dapi_path)
        self.np_path = Path(np_path)

        dapi = self._load_grayscale_tiff(self.dapi_path, "DAPI")
        np_image = self._load_grayscale_tiff(self.np_path, "nanoparticle")

        if dapi.shape != np_image.shape:
            raise ValueError(
                "DAPI and nanoparticle images must have identical dimensions "
                "and be spatially aligned. "
                f"DAPI shape: {dapi.shape}; NP shape: {np_image.shape}."
            )

        self.dapi = dapi
        self.np_image = np_image

        # Derive a common output stem from names such as field1_dapi.tif and
        # field1_np.tif. If no suffix is present, use the DAPI filename stem.
        stem = self.dapi_path.stem
        lower_stem = stem.lower()
        if lower_stem.endswith("_dapi"):
            stem = stem[:-5]
        elif lower_stem.endswith("-dapi"):
            stem = stem[:-5]
        self.output_stem = stem or self.dapi_path.stem
        return dapi, np_image

    # DAPI segmentation
    def segment_nuclei(self, dapi=None):
        """
        Segment DAPI-positive nuclei only.

        Uses broad-background subtraction, a permissive Otsu-derived
        threshold, and distance-transform watershed to help separate
        touching nuclei. The NP image is never used in this function.
        """
        if dapi is None:
            if self.dapi is None:
                raise RuntimeError("Load the DAPI image before segmentation.")
            dapi = self.dapi

        dapi = np.asarray(dapi, dtype=np.float32)
        if dapi.ndim != 2:
            raise ValueError("DAPI segmentation expects a 2D image.")

        smooth = filters.gaussian(dapi, sigma=1, preserve_range=True)
        broad_background = filters.gaussian(
            smooth,
            sigma=self.dapi_background_sigma,
            preserve_range=True,
        )
        corrected = smooth - broad_background
        corrected[corrected < 0] = 0
        self.dapi_corrected = corrected

        positive = corrected[corrected > 0]
        if positive.size == 0 or float(positive.max()) == 0:
            self.dapi_threshold = 0.0
            self.dapi_binary = np.zeros_like(corrected, dtype=bool)
            self.nuclei = np.zeros_like(corrected, dtype=np.int32)
            return self.nuclei

        try:
            otsu_threshold = float(filters.threshold_otsu(positive))
        except (ValueError, RuntimeError):
            otsu_threshold = float(np.percentile(positive, 50))

        # Otsu is scaled down slightly to retain dim nuclear edges.
        threshold = max(0.0, otsu_threshold * self.threshold_factor)
        if threshold == 0:
            threshold = float(np.finfo(np.float32).eps)
        binary = corrected > threshold

        # Remove only small specks; fill small holes inside nuclei.
        binary = morphology.remove_small_objects(
            binary,
            min_size=self.min_object_size,
            connectivity=2,
        )
        binary = morphology.remove_small_holes(
            binary,
            area_threshold=200,
            connectivity=2,
        )
        self.dapi_threshold = threshold
        self.dapi_binary = binary.copy()

        if not np.any(binary):
            self.nuclei = np.zeros_like(corrected, dtype=np.int32)
            return self.nuclei

        # Watershed splits touching nuclei when distinct distance peaks exist.
        distance = ndimage.distance_transform_edt(binary)
        coordinates = peak_local_max(
            distance,
            min_distance=max(1, self.watershed_min_distance),
            labels=binary.astype(np.uint8),
            exclude_border=False,
        )
        markers = np.zeros(binary.shape, dtype=np.int32)
        for marker_id, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = marker_id

        if markers.max() == 0:
            candidate_labels = measure.label(binary, connectivity=2)
        else:
            candidate_labels = watershed(
                -distance,
                markers,
                mask=binary,
                watershed_line=False,
            )

        filtered = np.zeros(candidate_labels.shape, dtype=np.int32)
        next_label = 1

        for region in measure.regionprops(candidate_labels):
            area = int(region.area)
            perimeter = float(region.perimeter)
            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0.0

            area_ok = self.min_nucleus_area <= area <= self.max_nucleus_area
            circularity_ok = (
                self.min_circularity <= 0
                or circularity >= self.min_circularity
            )

            if area_ok and circularity_ok:
                filtered[candidate_labels == region.label] = next_label
                next_label += 1

        self.nuclei = filtered
        return filtered

    # NP background correction
    def subtract_background(self, fluorescence=None):
        """
        Estimate and subtract slowly varying background from the NP image.
        This step does not use DAPI intensities.
        """
        if fluorescence is None:
            if self.np_image is None:
                raise RuntimeError("Load the nanoparticle image first.")
            fluorescence = self.np_image

        fluorescence = np.asarray(fluorescence, dtype=np.float32)
        if fluorescence.ndim != 2:
            raise ValueError("Fluorescence correction expects a 2D image.")

        background = filters.gaussian(
            fluorescence,
            sigma=self.background_sigma,
            preserve_range=True,
        )
        corrected = fluorescence - background
        corrected[corrected < 0] = 0

        self.background = background
        self.corrected_fluorescence = corrected
        return corrected

    # Measurement regions
    def create_cell_territories(self):
        """
        Expand each nucleus outward by cell_radius pixels.

        Pixels in overlapping territories are assigned to the nearest
        nucleus.
        """
        if self.nuclei is None:
            raise RuntimeError("Nuclei have not been segmented.")

        labels = self.nuclei
        all_nuclei = labels > 0
        territories = np.zeros_like(labels, dtype=np.int32)

        if not np.any(all_nuclei):
            self.cell_territories = territories
            return territories

        distance, nearest_indices = ndimage.distance_transform_edt(
            ~all_nuclei,
            return_indices=True,
        )
        nearest_labels = labels[nearest_indices[0], nearest_indices[1]]
        
        valid = distance <= self.cell_radius
        territories[valid] = nearest_labels[valid]

        self.cell_territories = territories
        return territories

    # Cell-level measurements
    def quantify_cells(self):
        """Measure corrected NP fluorescence within each cell territory."""
        if self.nuclei is None:
            raise RuntimeError("Nuclei have not been segmented.")
        if self.corrected_fluorescence is None:
            raise RuntimeError("NP fluorescence has not been background-subtracted.")

        territories = self.create_cell_territories()
        measurements = []

        for region in measure.regionprops(self.nuclei):
            cell_id = int(region.label)
            cell_mask = territories == cell_id
            values = self.corrected_fluorescence[cell_mask]

            if values.size:
                mean_value = float(np.mean(values))
                integrated_value = float(np.sum(values))
            else:
                mean_value = np.nan
                integrated_value = np.nan

            measurements.append(
                {
                    "cell_id": cell_id,
                    "nucleus_area_px": int(region.area),
                    "cell_area_px": int(cell_mask.sum()),
                    "mean_cell_fluorescence": mean_value,
                    "integrated_cell_fluorescence": integrated_value,
                }
            )

        self.results = pd.DataFrame(
            measurements,
            columns=[
                "cell_id",
                "nucleus_area_px",
                "cell_area_px",
                "mean_cell_fluorescence",
                "integrated_cell_fluorescence",
            ],
        )
        return self.results

    # Main analysis
    def analyze(self, dapi_path, np_path, output_dir=None, save_outputs=True):
        """
        Run analysis on a matched DAPI/NP image pair.

        Parameters
        ----------
        dapi_path : str or Path
            Grayscale DAPI TIFF, e.g. ``field1_dapi.tif``.
        np_path : str or Path
            Grayscale nanoparticle-fluorescence TIFF, e.g. ``field1_np.tif``.
        output_dir : str or Path, optional
            Directory for outputs. Defaults to the DAPI image directory.
        save_outputs : bool
            If True, save CSV, masks, and QC images automatically.
        """
        self.load_images(dapi_path, np_path)

        self.output_dir = (
            Path(output_dir) if output_dir is not None else self.dapi_path.parent
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 68)
        print("NanoCyte Analysis")
        print("=" * 68)
        print(f"DAPI image: {self.dapi_path}")
        print(f"NP image:   {self.np_path}")
        print(f"Image dimensions: {self.dapi.shape[1]} x {self.dapi.shape[0]}")

        print("\n1. Segmenting DAPI nuclei...")
        nuclei = self.segment_nuclei()
        n_cells = int(nuclei.max())
        print(f"   Nuclei detected: {n_cells}")
        print(f"   DAPI threshold: {self.dapi_threshold:.4g}")

        print("\n2. Background-subtracting NP fluorescence...")
        self.subtract_background()

        print("\n3. Measuring fluorescence within cell territories...")
        print(f"   Outward radius: {self.cell_radius} pixels")
        results = self.quantify_cells()
        print(f"   Cells quantified: {len(results)}")

        if not results.empty:
            avg_fluorescence = results["mean_cell_fluorescence"].mean()
            print(f"   Average cell fluorescence: {avg_fluorescence:.3f}")
        else:
            print("   Average cell fluorescence: N/A (no cells quantified)")

        if save_outputs:
            self.save_all_outputs(self.output_dir)

        print("\nAnalysis complete.")
        return results

    # Output helpers
    def _require_analysis(self):
        if self.nuclei is None or self.dapi is None or self.np_image is None:
            raise RuntimeError("Run analyze() before saving output images.")
        if self.output_stem is None:
            raise RuntimeError("No source image pair is associated with the analysis.")

    def _get_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = self.dapi_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _display_scaled(image, low=1, high=99.8):
        """Return percentile-scaled data for display only."""
        image = np.asarray(image, dtype=np.float32)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image)
        vmin, vmax = np.percentile(finite, [low, high])
        if vmax <= vmin:
            vmax = vmin + 1.0
        return np.clip((image - vmin) / (vmax - vmin), 0, 1)

    def save_results(self, output_dir=None):
        """Save the cell-level fluorescence table as CSV."""
        if self.results is None:
            raise RuntimeError("Run analyze() before saving results.")
        output_dir = self._get_output_dir(output_dir)
        path = output_dir / f"{self.output_stem}_cell_associated_nanoparticle_fluorescence.csv"
        self.results.to_csv(path, index=False)
        print(f"Saved CSV: {path}")
        return path

    def save_nucleus_segmentation(self, output_dir=None):
        """Save DAPI and the segmented nucleus labels for visual inspection."""
        self._require_analysis()
        output_dir = self._get_output_dir(output_dir)
        path = output_dir / f"{self.output_stem}_dapi_segmentation.png"

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].imshow(self._display_scaled(self.dapi), cmap="gray")
        axes[0].set_title("DAPI image")
        axes[0].axis("off")

        axes[1].imshow(self._display_scaled(self.dapi), cmap="gray")
        axes[1].imshow(
            np.ma.masked_where(self.nuclei == 0, self.nuclei),
            cmap="nipy_spectral",
            alpha=0.65,
            interpolation="nearest",
        )
        axes[1].set_title(f"Detected nuclei (n={int(self.nuclei.max())})")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved DAPI segmentation: {path}")
        return path

    def save_nucleus_mask(self, output_dir=None):
        """Save the labeled nucleus mask as a TIFF (0 = background)."""
        self._require_analysis()
        output_dir = self._get_output_dir(output_dir)
        path = output_dir / f"{self.output_stem}_nucleus_mask.tif"
        tiff.imwrite(path, self.nuclei.astype(np.uint32))
        print(f"Saved labeled nucleus mask: {path}")
        return path

    def save_radius_overlay(self, output_dir=None, label_cells=True, font_size=6):
        """Overlay nucleus outlines and measurement-territory boundaries on NP image."""
        self._require_analysis()
        output_dir = self._get_output_dir(output_dir)
        path = output_dir / f"{self.output_stem}_measurement_radii.png"

        territories = self.create_cell_territories()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self._display_scaled(self.np_image), cmap="gray")

        # Territory boundaries: the border of each cell's assigned territory.
        for region in measure.regionprops(self.nuclei):
            label = region.label
            nucleus_mask = self.nuclei == label
            cell_mask = territories == label

            nucleus_boundary = morphology.binary_dilation(nucleus_mask) & ~nucleus_mask
            cell_boundary = morphology.binary_dilation(cell_mask) ^ cell_mask

            yn, xn = np.where(nucleus_boundary)
            if yn.size:
                ax.scatter(xn, yn, s=0.25, alpha=0.9)

            yc, xc = np.where(cell_boundary)
            if yc.size:
                ax.scatter(xc, yc, s=0.25, alpha=0.7)

            if label_cells:
                cy, cx = region.centroid
                ax.text(
                    cx, cy, str(label),
                    fontsize=font_size,
                    ha="center", va="center",
                    color="yellow",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=0.5),
                )

        ax.set_title("NP image with DAPI nuclei and assigned measurement territories")
        ax.axis("off")
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved measurement-region overlay: {path}")
        return path


    def save_combined_overlay(self, output_dir=None):
        """Save a composite DAPI + NP overlay with nucleus and measurement-region outlines."""

        self._require_analysis()
        output_dir = self._get_output_dir(output_dir)
        path = output_dir / f"{self.output_stem}_combined_overlay.png"

        dapi_scaled = self._display_scaled(self.dapi)
        np_scaled = self._display_scaled(self.corrected_fluorescence)

        # DAPI in blue and nanoparticle fluorescence in red.
        rgb = np.zeros((*self.dapi.shape, 3), dtype=np.float32)
        rgb[..., 2] = dapi_scaled
        rgb[..., 0] = np_scaled

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)

        # Generate the same measurement territories used by save_radius_overlay().
        territories = self.create_cell_territories()

        for region in measure.regionprops(self.nuclei):
            label = region.label

            # Outline the nucleus in cyan.
            nucleus_mask = self.nuclei == label
            nucleus_boundary = (
                morphology.binary_dilation(nucleus_mask) & ~nucleus_mask
            )
            yn, xn = np.where(nucleus_boundary)
            if yn.size:
                ax.scatter(xn, yn, s=0.1, c="cyan", alpha=0.5)

            # Outline the assigned measurement territory.
            cell_mask = territories == label
            cell_boundary = morphology.binary_dilation(cell_mask) ^ cell_mask
            yc, xc = np.where(cell_boundary)
            if yc.size:
                ax.scatter(xc, yc, s=0.1, c="yellow", alpha=0.5)

        ax.set_title("DAPI (blue) + NP (red) + measurement regions (yellow)")
        ax.axis("off")

        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"Saved combined overlay: {path}")
        return path

    def save_all_outputs(self, output_dir=None):
        """Save all CSV, mask, and visual quality-control outputs."""
        self._require_analysis()
        output_dir = self._get_output_dir(output_dir)
        paths = [
            self.save_results(output_dir),
            self.save_nucleus_segmentation(output_dir),
            self.save_nucleus_mask(output_dir),
            self.save_radius_overlay(output_dir),
            self.save_combined_overlay(output_dir),
        ]
        return paths

    def print_summary(self):
        """Print summary statistics for per-cell mean fluorescence."""
        if self.results is None:
            raise RuntimeError("Run analyze() before printing a summary.")

        values = self.results["mean_cell_fluorescence"].dropna()
        print("\n" + "=" * 68)
        print("SUMMARY")
        print("=" * 68)
        print(f"Cells analyzed: {len(values)}")
        print(f"Outward measurement radius: {self.cell_radius} px")
        if values.empty:
            print("No cells had measurable territory pixels.")
        else:
            print(f"Mean per-cell fluorescence:   {values.mean():.3f}")
            print(f"SD per-cell fluorescence:     {values.std():.3f}")
            print(f"Minimum:                      {values.min():.3f}")
            print(f"Maximum:                      {values.max():.3f}")
        print("=" * 68)


if __name__ == "__main__":
    # Edit these paths to point to one matched image pair before running
    # this file directly. The GUI can call the same class later.
    dapi_file = Path("field1_dapi.tif")
    np_file = Path("field1_np.tif")

    analyzer = CellFluorescenceAnalyzer(
        min_nucleus_area=150,
        max_nucleus_area=5000,
        min_circularity=0.25,
        cell_radius=20,
        threshold_factor=0.05,
    )
    analyzer.analyze(dapi_file, np_file)
    analyzer.print_summary()

