"""KiriSMILES database (kdb): records for a Materials-Project-like store of kirigami
designs, specimens and results.

Four kinds of record, each a JSON document validated by the models below:

  Material   constitutive law and constants (E, nu, model, density), with a source.
             Shared by many specimens. Thickness is not here: it belongs to the specimen.
  Design     what is cut, with no unit and no sheet: a canonical KiriSMILES string,
             or, for patterns the notation cannot write yet (tube kirigami, explicit
             geometry), a generator with its parameters. The design_id is a hash of
             that description, so it is scale-free.
  Specimen   one design realised: unit of length, sheet (flat L x W, or a cylinder),
             thickness, material, fabrication details. Carries the geometry key of
             the finite pattern (kirismiles.geometry_key), which depends on the sheet.
  Result     one simulation, experiment or digitised curve on a specimen: loading and
             boundary conditions, method and settings, links to raw data, derived
             properties (dimensional and dimensionless), provenance.

Raw data (npz fields, JSON curves, GIFs) is linked, not stored: Result.data maps a
name to a DataRef with the path, size, sha256, format and, for npz, every array's
shape, dtype and meaning, plus an optional URL of an archived copy.

A design has many specimens, a specimen many results. IDs are content hashes, so
rebuilding the database from the same sources gives the same IDs.

  python -m kirigami.kdb schema [out_dir]     write the JSON Schema of each record
"""
import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "kdb/0.2"     # 0.2: raw data as DataRef (path, sha256, arrays); 0.1 paths still read


def hid(prefix, obj, n=12):
    """Content hash ID: prefix + first n hex digits of sha1(canonical JSON)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha1(s.encode()).hexdigest()[:n]}"


class _Rec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Optional[float]
    unit: str = Field(description="'1' for dimensionless")


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doi: Optional[str] = None
    citation_key: Optional[str] = Field(None, description="key in paper/references.bib")
    figure: Optional[str] = None
    role: Literal["origin", "data", "constants", "method", "other"] = "other"


# ------------------------------------------------------------------ material
class Material(_Rec):
    material_id: str
    name: Optional[str] = None
    model: Literal["svk", "neo_hookean", "linear"] = Field(
        "svk", description="membrane law of the shell model (kirigami/shell.py)")
    E: Quantity = Field(description="Young's modulus")
    nu: float = Field(description="Poisson's ratio")
    density: Optional[Quantity] = None
    discreteness_length: Optional[Quantity] = Field(
        None, description="smallest physical length of the material (e.g. the lattice spacing "
                          "of a 2D crystal); None for a continuum. Ligaments comparable to it "
                          "need an atomistic model")
    references: list[Reference] = []

    @staticmethod
    def make_id(E_MPa, nu, model, density=None):
        return hid("km", {"E": round(float(E_MPa), 6), "nu": round(float(nu), 6),
                          "model": model, "rho": density})


# -------------------------------------------------------------------- design
class Generator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="generator in kirigami/geometry.py or a script")
    params: dict = Field(description="lengths in the specimen's unit are not allowed here "
                                     "unless the design is tied to a scale (scale_free=False)")


class Design(_Rec):
    design_id: str
    kirismiles: Optional[str] = Field(None, description="canonical KiriSMILES string")
    generator: Optional[Generator] = Field(
        None, description="when no string: how the pattern is generated")
    scale_free: bool = Field(
        True, description="False when the identity depends on the sheet (explicit geometry)")
    cuts_as_slots: bool = Field(
        False, description="the cuts must have a finite width (e.g. where cuts meet); the width "
                           "itself is fabrication and is the specimen's kerf")
    family: Optional[str] = None
    references: list[Reference] = []

    @staticmethod
    def make_id(kirismiles=None, generator=None, geometry_key=None):
        if kirismiles:
            return hid("kd", {"kirismiles": kirismiles})
        if generator is not None:
            return hid("kd", {"generator": generator})
        return hid("kd", {"geometry_key": geometry_key})


# ------------------------------------------------------------------ specimen
class FlatSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["flat"] = "flat"
    L: float = Field(gt=0, description="length along the pull (x)")
    W: float = Field(gt=0, description="width (y)")


class Cylinder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cylinder"] = "cylinder"
    circumference: float = Field(gt=0)
    length: Optional[float] = Field(None, gt=0, description="None when not reported")
    n_around: Optional[int] = Field(None, ge=1, description="unit cells around the tube")
    rows: Optional[int] = Field(None, ge=1, description="unit cells along the tube")
    rolled_from_flat: bool = Field(True, description="natural shape flat (rolled sheet) "
                                                     "or curved (cut from tubing)")


class Specimen(_Rec):
    specimen_id: str
    design_id: str
    unit: str = Field("mm", description="unit of every length of this specimen and its design")
    geometry: FlatSheet | Cylinder = Field(discriminator="type")
    thickness: float = Field(gt=0)
    material_id: str
    kerf: Optional[float] = Field(
        None, ge=0, description="fabrication cut width, applied to every cut; None = zero-width cuts "
                          "or not reported. Widths chosen as geometry stay in the string")
    geometry_key: Optional[str] = Field(None, description="kirismiles.geometry_key of the "
                                                          "finite pattern on this sheet")
    dimensionless: dict[str, float] = {}

    @staticmethod
    def make_id(design_id, unit, geometry, thickness, material_id, kerf):
        return hid("ks", {"d": design_id, "u": unit, "g": geometry, "h": round(float(thickness), 6),
                          "m": material_id, "k": kerf})


# -------------------------------------------------------------------- result
class Loading(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["uniaxial", "biaxial"] = "uniaxial"
    boundary: Literal["clamped", "roller", "biaxial", "periodic"] = "clamped"
    in_plane: bool = False
    max_displacement: Optional[float] = None
    max_strain: Optional[float] = None
    steps: Optional[int] = None


class Method(BaseModel):
    model_config = ConfigDict(extra="allow")
    engine: Optional[str] = Field(None, description="'kirigami' for this platform; or the "
                                                    "code/apparatus of a source")
    model: Optional[str] = Field(None, description="e.g. 'Kirchhoff-Love shell'")
    analysis: Optional[str] = Field(None, description="simulate, buckle, ...")
    mesh: Optional[dict] = None
    code_version: Optional[str] = None
    image: Optional[str] = None
    seconds: Optional[float] = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Optional[str] = Field(None, description="path of the run or file it came from")
    created: Optional[str] = None
    approximate: bool = Field(False, description="True for digitised curves")
    references: list[Reference] = []
    notes: Optional[str] = None


class ArrayInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: list[int]
    dtype: str
    description: Optional[str] = None


class DataRef(BaseModel):
    """A raw-data file that a result links to. The file is not copied into the
    database: the record says where it is, what is in it, and how to check that a
    copy is the original (sha256)."""
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="path relative to the repository root")
    key: Optional[str] = Field(None, description="entry inside the file, e.g. one curve of a "
                                                 "JSON file holding several")
    format: Optional[str] = Field(None, description="npz, json, csv, gif, png, ...")
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    arrays: dict[str, ArrayInfo] = Field({}, description="npz: name -> shape, dtype, meaning")
    description: Optional[str] = None
    url: Optional[str] = Field(None, description="archived copy (e.g. a Zenodo DOI or a "
                                                 "storage URL), when the file is published")
    available: bool = Field(True, description="False when the file was missing at build time")


def data_ref(path, root=".", key=None, description=None, array_descriptions=None, url=None):
    """DataRef for a file: size, sha256 and, for npz, the name, shape and dtype of every
    array (read from the .npy headers, without loading the arrays)."""
    import zipfile
    from pathlib import Path
    import numpy as np
    f = Path(root) / path
    ref = dict(path=str(path).replace("\\", "/"), key=key, format=f.suffix.lstrip(".") or None,
               description=description, url=url)
    if not f.is_file():
        return DataRef(**ref, available=False)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    ref.update(bytes=f.stat().st_size, sha256=h.hexdigest())
    if f.suffix == ".npz":
        desc = array_descriptions or {}
        arrays = {}
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                with z.open(name) as fh:
                    version = np.lib.format.read_magic(fh)
                    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                            else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = read(fh)
                a = name[:-4] if name.endswith(".npy") else name
                d = desc.get(a)
                if d is None:                   # pattern entries such as "eps*"
                    d = next((v for k, v in desc.items() if k.endswith("*") and a.startswith(k[:-1])),
                             None)
                arrays[a] = ArrayInfo(shape=list(shape), dtype=str(dtype), description=d)
        ref["arrays"] = arrays
    return DataRef(**ref)


class Result(_Rec):
    result_id: str
    specimen_id: str
    kind: Literal["simulation", "experiment", "digitised"]
    status: Optional[str] = None
    completed: Optional[bool] = None
    loading: Loading
    method: Method
    data: dict[str, DataRef] = Field({}, description="name -> raw-data file (linked, not stored)")
    properties: dict[str, Quantity] = {}
    provenance: Provenance

    @field_validator("data", mode="before")
    @classmethod
    def _paths_to_refs(cls, v):
        """Records of kdb/0.1 gave each raw-data file as a bare path ('file#key'
        for an entry inside it)."""
        out = {}
        for name, r in (v or {}).items():
            if isinstance(r, str):
                p, _, k = r.partition("#")
                r = {"path": p, "key": k or None}
            out[name] = r
        return out


RECORDS = {"material": Material, "design": Design, "specimen": Specimen, "result": Result}


def write_schemas(out="schemas/kdb"):
    from pathlib import Path
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    for name, M in RECORDS.items():
        s = M.model_json_schema()
        s["$id"] = f"{SCHEMA_VERSION}/{name}"
        (p / f"{name}.schema.json").write_text(json.dumps(s, indent=1))
    return [str(p / f"{n}.schema.json") for n in RECORDS]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "schema":
        print("\n".join(write_schemas(*sys.argv[2:3])))
    else:
        print(__doc__)
