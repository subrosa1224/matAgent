"""Read-only rendering in unrotated PDF points; original rotation is retained."""

import math

from .figure_evidence_contracts import PageRegion


def screen_region(region, *, display_width, display_height, page_width, page_height):
    """Map a rectangle on the exact unrotated source image, not a browser viewport."""
    region = PageRegion.model_validate(region.model_dump())
    values = (display_width, display_height, page_width, page_height)
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Invalid display transform")
    if region.x1 > display_width or region.y1 > display_height:
        raise ValueError("Screen region outside displayed image")
    return PageRegion(
        x0=region.x0 * page_width / display_width,
        x1=region.x1 * page_width / display_width,
        y0=region.y0 * page_height / display_height,
        y1=region.y1 * page_height / display_height,
    )


def render_source(path, *, page, region, store, cancel_event=None):
    import pymupdf

    if type(page) is not int or page < 1:
        raise ValueError("Invalid PDF page")
    region = PageRegion.model_validate(region.model_dump())
    with pymupdf.open(path) as document:
        if page > len(document) or document.is_encrypted:
            raise ValueError("PDF page unavailable")
        source = document[page - 1]
        rotation = source.rotation
        source.set_rotation(0)  # in-memory only; never save the source document
        bounds = source.rect
        if region.x1 > bounds.width or region.y1 > bounds.height:
            raise ValueError("Region outside PDF page")
        images = []
        for rect, dpi in (
            (bounds, 150),
            (pymupdf.Rect(region.x0, region.y0, region.x1, region.y1), 300),
        ):
            if cancel_event is not None and cancel_event.is_set():
                raise ValueError("Figure rendering cancelled")
            if (math.ceil(rect.width * dpi / 72) + 1) * (
                math.ceil(rect.height * dpi / 72) + 1
            ) > 4_000_000:
                raise ValueError("Render pixel budget exceeded")
            image = source.get_pixmap(dpi=dpi, clip=rect, alpha=False)
            images.append(store.save_image(image.tobytes("png")))
        return dict(
            page_width=bounds.width,
            page_height=bounds.height,
            original_rotation=rotation,
            renderer_version="PyMuPDF-" + pymupdf.VersionBind,
            page_image=images[0],
            crop_image=images[1],
        )


def render_page(path, *, page, store):
    import pymupdf

    if type(page) is not int or page < 1:
        raise ValueError("Invalid PDF page")
    with pymupdf.open(path) as document:
        if page > len(document) or document.is_encrypted:
            raise ValueError("PDF page unavailable")
        source = document[page - 1]
        rotation = source.rotation
        source.set_rotation(0)
        rect = source.rect
        if (math.ceil(rect.width * 150 / 72) + 1) * (
            math.ceil(rect.height * 150 / 72) + 1
        ) > 4_000_000:
            raise ValueError("Render pixel budget exceeded")
        image = source.get_pixmap(dpi=150, alpha=False)
        return dict(
            page_width=rect.width,
            page_height=rect.height,
            original_rotation=rotation,
            renderer_version="PyMuPDF-" + pymupdf.VersionBind,
            image=store.save_image(image.tobytes("png")),
        )
