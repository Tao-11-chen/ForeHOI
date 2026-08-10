"""Minimal nvdiffrast mesh renderer for the feature-matching pose stage.

Renders RGB + camera-space depth of a trimesh mesh given an OpenCV-convention
object-to-camera pose (x right, y down, z forward) and an intrinsics matrix.
Only what MASt3R matching + correspondence lifting need — no lighting model,
no anti-aliasing, no mipmaps.

Appearance priority: base-color texture > vertex colors > lambertian gray from
vertex normals (fallback so that even geometry-only meshes remain matchable).
"""
import numpy as np
import torch
import nvdiffrast.torch as dr


def projection_from_intrinsics(K, width, height, znear, zfar):
    """OpenGL clip-space projection for an OpenCV camera (y-down image coords),
    so rasterized pixels line up with image pixel (row, col) indexing."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    skew = K[0, 1]
    q = -(zfar + znear) / (zfar - znear)
    qn = -2.0 * zfar * znear / (zfar - znear)
    return np.array([
        [2.0 * fx / width, -2.0 * skew / width, (-2.0 * cx + width) / width, 0.0],
        [0.0, 2.0 * fy / height, (-2.0 * cy + height) / height, 0.0],
        [0.0, 0.0, q, qn],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)


class MeshRenderer:
    """Bakes a (metrically scaled) mesh once, then renders it under arbitrary poses."""

    def __init__(self, device="cuda"):
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device)
        self.center = None   # (3,) bbox center that was subtracted from the vertices
        self.diag = None     # bbox diagonal of the (scaled) mesh, in metric units
        self._v = None       # (V,3) float32 centered vertices
        self._f = None       # (F,3) int32 faces
        self._appearance = None

    def set_mesh(self, mesh):
        """Bake vertices/faces/appearance to device tensors.

        Vertices are stored centered on the bbox center; poses produced with this
        renderer are therefore object-to-camera transforms of the CENTERED mesh
        (convert back with `pose @ translate(-center)`, see estimator).
        """
        import trimesh  # local import: keeps module importable without trimesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise TypeError("MeshRenderer needs a trimesh.Trimesh with faces")

        v = np.asarray(mesh.vertices, dtype=np.float32)
        vmin, vmax = v.min(axis=0), v.max(axis=0)
        self.center = ((vmin + vmax) / 2.0).astype(np.float32)
        self.diag = float(np.linalg.norm(vmax - vmin))
        self._v = torch.from_numpy(v - self.center).to(self.device)
        self._f = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(torch.int32).to(self.device).contiguous()
        self._appearance = self._bake_appearance(mesh)

    def _bake_appearance(self, mesh):
        """Return one of:
        ("texture", uv (V,2) float32 tensor, tex (1,h,w,3) float32 tensor)
        ("vertex",  rgb (V,3) float32 tensor)
        ("normal",  nrm (V,3) float32 tensor)
        """
        visual = getattr(mesh, "visual", None)
        try:
            kind = getattr(visual, "kind", None)
            if kind == "texture":
                uv = np.asarray(visual.uv, dtype=np.float32)
                material = visual.material
                # explicit None checks: a numpy-array texture would make `or` raise
                img = getattr(material, "baseColorTexture", None)
                if img is None:
                    img = getattr(material, "image", None)
                if uv is not None and len(uv) == len(mesh.vertices) and img is not None:
                    tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                    # trimesh keeps glTF-convention UVs (v=0 at the top row); dr.texture
                    # with a (1,h,w,c) tensor samples v=0 at row 0, so no flip is needed.
                    return ("texture",
                            torch.from_numpy(uv).to(self.device),
                            torch.from_numpy(tex[None]).to(self.device).contiguous())
            if kind == "vertex":
                vc = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
                if len(vc) == len(mesh.vertices):
                    return ("vertex", torch.from_numpy(vc).to(self.device))
        except Exception:
            pass  # fall through to normal shading
        nrm = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return ("normal", torch.from_numpy(nrm).to(self.device))

    def render(self, K, ob_in_cam, height=518, width=518, znear=None, zfar=None):
        """Render the mesh. Returns (rgb (H,W,3) uint8, depth (H,W) float32).

        depth is the camera-space z of the visible surface (0 where background),
        suitable for lifting pixels to 3D via the inverse intrinsics.
        """
        assert self._v is not None, "call set_mesh() first"
        ob_in_cam = np.asarray(ob_in_cam, dtype=np.float32)
        dist = float(np.linalg.norm(ob_in_cam[:3, 3]))
        if znear is None:
            znear = max(0.01, dist - 3.0 * self.diag)
        if zfar is None:
            zfar = dist + 3.0 * self.diag

        rot = torch.from_numpy(ob_in_cam[:3, :3]).to(self.device)
        trans = torch.from_numpy(ob_in_cam[:3, 3]).to(self.device)
        v_cam = self._v @ rot.T + trans                        # (V,3), z forward
        proj = torch.from_numpy(projection_from_intrinsics(K, width, height, znear, zfar)).to(self.device)
        v_clip = torch.cat([v_cam, torch.ones_like(v_cam[:, :1])], dim=-1) @ proj.T
        rast, _ = dr.rasterize(self.glctx, v_clip[None].contiguous(), self._f, (height, width))
        hit = rast[0, ..., 3] > 0                             # (H,W) triangle-id mask

        zbuf, _ = dr.interpolate(v_cam[:, 2:3].contiguous()[None], rast, self._f)
        depth = torch.where(hit, zbuf[0, ..., 0], torch.zeros((), device=self.device))

        mode, attr = self._appearance[0], self._appearance[1:]
        if mode == "texture":
            uv, tex = attr
            uv_i, _ = dr.interpolate(uv[None], rast, self._f)
            color = dr.texture(tex, uv_i, filter_mode="linear", boundary_mode="clamp")[0]
        elif mode == "vertex":
            color, _ = dr.interpolate(attr[0][None].contiguous(), rast, self._f)
            color = color[0]
        else:  # lambertian gray from interpolated vertex normals
            nrm, _ = dr.interpolate(attr[0][None].contiguous(), rast, self._f)
            nrm = torch.nn.functional.normalize(nrm[0], dim=-1)
            light = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # headlight
            lam = nrm @ light
            color = (0.25 + 0.75 * lam.abs())[..., None].expand(-1, -1, 3)

        color = torch.where(hit[..., None], color.clamp(0.0, 1.0), torch.zeros((), device=self.device))
        rgb = (color * 255.0).round().to(torch.uint8).cpu().numpy()
        return rgb, depth.cpu().numpy().astype(np.float32)
