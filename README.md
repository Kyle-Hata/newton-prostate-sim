# newton-prostate-sim
Deformable organ simulation pipeline
# newton-prostate-sim

Author: Kyle Hata (Brigham and Women's Hospital, Department of Radiology)

This project simulates pelvic organs (prostate, bladder, rectum, urethra) as
deformable soft bodies, so that a surgical tool can interact with them and the
surrounding organs respond. Organ geometry comes from an MRI scan segmented in 3D
Slicer; the tissue is simulated with the NVIDIA Newton/Warp physics engine and
the results are viewed in NVIDIA Isaac Sim. The long-term goal is tool-tissue
interaction (poking, and eventually cutting) with realistic deformation of the
surrounding anatomy.

Pipeline: 3D Slicer (segmentation) -> TetGen (tetrahedral mesh) -> Newton/Warp
(soft-body simulation) -> USD file -> Isaac Sim (viewing).

## Requirements

- A CUDA GPU with 16 GB of memory (tested on RTX 4060 Ti / 5060 Ti). Sessions
  were run on rented Vast.ai instances using the "PyTorch (Vast)" template.
- NVIDIA Isaac Sim on a local machine, for viewing the recorded `.usd` output.
- 3D Slicer, for segmenting the scan and exporting organ meshes.

## Installation

Clone the repository and run the setup script, which installs Newton, TetGen, and
PyVista and verifies that the GPU is detected:

```
$ git clone https://github.com/Kyle-Hata/newton-prostate-sim.git
$ cd newton-prostate-sim
$ bash setup.sh
```

Wait for the message confirming the RTX GPU was detected. To save work back to
the repository, set your git identity on a fresh machine:

```
$ git config --global user.email "khata@terpmail.umd.edu"
$ git config --global user.name "Kyle-Hata"
```

## Preparing organ meshes

The organs are segmented from an MRI scan in 3D Slicer into separate structures
(prostate, bladder, rectum, urethra, bone) and exported as STL surface meshes via
the Segmentations module (Export to files). Because every organ is segmented from
the same scan, the exported meshes already share one coordinate system and stay
in their correct relative positions.

Convert each organ's surface mesh into a tetrahedral (volume) mesh, which is what
the soft-body solver requires. `make_tets_multi.py` takes the organ name and
preserves the real millimeter coordinates, so positions are kept:

```
$ python make_tets_multi.py prostate_NEW
$ python make_tets_multi.py bladder_NEW
```

Each run writes `tet_nodes_<name>.npy` and `tet_elems_<name>.npy`.

## Running a simulation

`load_all_organs.py` loads the four soft organs into one scene, runs the Newton
VBD solver (with gravity disabled, since the organs are held in place by
surrounding tissue), and records the result to a USD file:

```
$ python load_all_organs.py
```

This writes `/workspace/all_organs.usd`. To include the pelvic bones as rigid
meshes, use `load_all_with_bones.py` instead.

## Viewing the result

The cloud machine has no display, so simulations are recorded to a `.usd` file and
viewed locally in Isaac Sim:

1. Download the `.usd` file from the Jupyter file browser.
2. In Isaac Sim: File -> Open, and select the `.usd` file.
3. Choose "Don't Save" on the empty-stage prompt.
4. Press Play, then press F to frame the view.

Isaac Sim is used only for viewing; all computation happens in Newton.

## Material parameters

Tissue stiffness is taken from published shear-wave elastography measurements and
converted to the solver's Lamé parameters using μ = E / (2(1+ν)) and
λ = Eν / ((1+ν)(1−2ν)), with Poisson's ratio ν = 0.45 (near-incompressible soft
tissue):

| Organ    | Young's modulus E | Reference |
|----------|-------------------|-----------|
| Prostate | ~30 kPa           | https://pmc.ncbi.nlm.nih.gov/articles/PMC4023053/ |
| Bladder  | ~17 kPa           | https://pubmed.ncbi.nlm.nih.gov/20443465/ |
| Urethra  | ~30.7 kPa         | https://www.mdpi.com/2306-5354/12/4/373 |
| Rectum   | ~1-3 kPa          | https://pmc.ncbi.nlm.nih.gov/articles/PMC8391413/ |
| Bone     | rigid (~GPa)      | treated as rigid, not simulated as soft tissue |

These are population-average literature values and are not validated for a
specific patient.

## Repository contents

Active pipeline:
- `setup.sh` — installs Newton, TetGen, and PyVista, and verifies the GPU.
- `make_tets_multi.py` — tetrahedralizes one organ STL, keeping real mm coordinates.
- `load_all_organs.py` — loads the four soft organs into one scene and records a USD file.
- `load_all_with_bones.py` — the same scene with the pelvic bones added as rigid meshes.

Milestones:
- `poke_sim.py` — the single-prostate poke (stable, USD output).
- `check_two_organs.py` — prostate and bladder stable together in one scene.
- `poke_two_organs.py` — prostate poked alongside the bladder (no contact between them).

Inter-organ contact:
- `push_into_bladder.py` — first interaction attempt.
- `push_contact_v2.py` — corrected contact radius for millimeter-scale meshes.
- `overlap_test.py` — forced-overlap test demonstrating the bladder responding to the prostate.

Surgical tool (in progress):
- `needle_indent.py`, `needle_v2.py` — rigid-needle indentation attempts.

Earlier and supporting scripts, retained for reference:
- `make_tets.py` — the original single-organ tetrahedralizer.
- `mycube.py`, `mytet.py`, `mysphere.py` — early soft-body building blocks.
- `check_mesh.py`, `check_tets.py`, `check_poke_moved.py`, `check_anchored.py`,
  `check_nogravity.py`, `check_poke.py`, `check_result.py`, `debug_sim.py` —
  diagnostic scripts that print mesh and node data.
- `drop.usd`, `gift.usd`, `squish.usd` — early USD recordings from Newton example
  soft bodies.

## Current status and limitations

The segmentation-to-simulation pipeline works and is verified: the four soft
organs simulate together stably, and a forced-overlap test confirms that force
transmits between organs (the bladder deforms in response to the prostate). Open
items: rigid-tool (needle) contact does not yet engage; the mesh (millimeter) and
stiffness (meter-based) units are not yet reconciled, so results are qualitatively
tissue-like but not a calibrated quantitative model; the deformation has not been
validated against ground-truth data; and cutting is not yet implemented, as it
requires mesh separation (e.g. the material point method) rather than the current
finite-element approach.
