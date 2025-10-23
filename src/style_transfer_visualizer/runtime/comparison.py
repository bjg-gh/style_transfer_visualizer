"""Helpers for building comparison images outside the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from style_transfer_visualizer.constants import COLOR_GREY
from style_transfer_visualizer.gallery import (
    ComparisonRenderOptions,
    render_comparison,
)
from style_transfer_visualizer.image_grid.naming import default_comparison_name
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.runtime.output import (
    stylized_image_path_from_paths,
)

if TYPE_CHECKING:  # pragma: no cover
    from style_transfer_visualizer.type_defs import LayoutName

__all__ = [
    "ComparisonRequest",
    "comparison_output_path",
    "render_comparison_image",
    "render_requested_comparisons",
]


@dataclass(slots=True)
class ComparisonRequest:
    """Bundle of comparison rendering options."""

    include_inputs: bool
    include_result: bool
    result_path: Path | None = None


def comparison_output_path(
    output_dir: Path | str,
    content_path: Path,
    style_path: Path,
    *,
    include_result: bool,
) -> Path:
    """
    Build the deterministic comparison name for the given inputs.

    The result variant appends ``_final`` to distinguish it from the
    inputs-only image rendered with the same stems.
    """
    out_dir = Path(output_dir)
    base = default_comparison_name(content_path, style_path, out_dir)
    if include_result:
        return base.parent / f"{base.stem}_final{base.suffix}"
    return base


def render_comparison_image(
    content_path: Path,
    style_path: Path,
    *,
    output_dir: Path | str,
    include_result: bool,
    result_path: Path | None = None,
) -> Path:
    """
    Render a gallery-style comparison image to the configured output dir.

    When ``include_result`` is true, ``result_path`` must point to an
    existing stylized image. The layout and canvas size mirror the CLI
    implementation so additional entry points can share the same look.
    """
    content_path = Path(content_path)
    style_path = Path(style_path)
    result_path = Path(result_path) if include_result and result_path else None

    with Image.open(content_path) as content_im:
        target_size = content_im.size

    layout: LayoutName = (
        "gallery-stacked-left" if include_result else "gallery-two-across"
    )
    out_path = comparison_output_path(
        output_dir, content_path, style_path, include_result=include_result,
    )

    return render_comparison(
        ComparisonRenderOptions(
            content_path=content_path,
            style_path=style_path,
            result_path=result_path,
            out_path=out_path,
            target_size=target_size,
            layout=layout,
            wall_color=COLOR_GREY,
            frame_style="gold",
            show_labels=True,
        ),
    )


def render_requested_comparisons(
    *,
    content_path: Path,
    style_path: Path,
    output_dir: Path | str,
    request: ComparisonRequest,
) -> list[Path]:
    """
    Render the comparison images requested by CLI or other entry points.

    Returns a list of paths that were written. When ``request.include_result``
    is true but the expected stylized image is missing, no file is written and
    a warning is logged.
    """
    output_dir = Path(output_dir)
    saved: list[Path] = []

    if request.include_inputs:
        saved.append(
            render_comparison_image(
                content_path=content_path,
                style_path=style_path,
                output_dir=output_dir,
                include_result=False,
            ),
        )

    if request.include_result:
        expected = (
            request.result_path
            if request.result_path is not None
            else stylized_image_path_from_paths(
                output_dir,
                content_path,
                style_path,
            )
        )
        if not expected.exists():
            logger.warning(
                "Expected stylized result missing: %s. "
                "Skipping content+style+result comparison.",
                expected,
            )
        else:
            saved.append(
                render_comparison_image(
                    content_path=content_path,
                    style_path=style_path,
                    output_dir=output_dir,
                    include_result=True,
                    result_path=expected,
                ),
            )

    return saved
