# GigaPose Custom Object Pipeline

This README documents the custom workflow in this repo for:

- onboarding one STL, or all STLs in a folder
- rendering GigaPose templates
- running GigaPose inference on a custom image set
- visualizing grasp points for one object at a time using a per-object `feasible_grasps.json`

This file is about the custom pipeline in this repo. The original upstream project notes are preserved in `README_gigapose_upstream.md`.

## Quick Start

If you want to verify the pipeline end to end before using your own assets, this repo includes a small demo bundle under `demo_asset/`. It is assumed that you are using the perseve-sam2 environment from Perseve's main repo, or have installed all relavent packages.

From the repository root:

```bash
conda activate perseve-sam2

export DATASET=hex_nut
export OBJ_ID=6
export STL_PATH=demo_asset/meshes/hex_nut.stl
export IMAGE_LIST=demo_asset/image_list.txt
export FEASIBLE_GRASPS_JSON=demo_asset/grasp_library/hex_nut_feasible_grasps.json
export MASK_ANNOTATION_DIR=demo_asset/mask_annotations

python -m src.scripts.prepare_single_object_img_list_dataset \
  --image-list $IMAGE_LIST \
  --mesh-path $STL_PATH \
  --dataset-name $DATASET \
  --dataset-root gigaPose_datasets/datasets \
  --obj-id $OBJ_ID

python -m src.scripts.render_templates_from_mesh_dir \
  gigaPose_datasets/datasets/$DATASET/models \
  --output-dir gigaPose_datasets/datasets/templates/$DATASET \
  --renderer blenderproc \
  --num-workers 1 \
  --num-gpus 1 \
  --disable-output \
  --mesh-scale 10

python test_myprints.py \
  --dataset-name $DATASET \
  --run-id $DATASET \
  --selected-obj-id $OBJ_ID \
  --test-setting localization

python -m src.scripts.adaptive_map_grasp_point_to_2d \
  --dataset-dir gigaPose_datasets/datasets/$DATASET \
  --result-csv gigaPose_datasets/results/large_$DATASET/predictions/large-pbrreal-rgb-mmodel_${DATASET}-test_${DATASET}.csv \
  --output-dir gigaPose_datasets/results/large_$DATASET/adaptive_grasp_point_mapping \
  --object-name $OBJ_ID \
  --feasible-grasps-json $FEASIBLE_GRASPS_JSON \
  --mask-annotation-dir $MASK_ANNOTATION_DIR
```

This demo bundle contains:

- one STL
- eight RGB frames
- matching semantic mask annotations for those frames
- one feasible grasp JSON for the demo object

## What This Pipeline Assumes

The custom pipeline expects:

- you cloned this repository from GitHub
- a conda environment named exactly `perseve-sam2`

```bash
conda activate perseve-sam2
```

- GigaPose checkpoint at:

```text
gigaPose_datasets/pretrained/gigaPose_v1.ckpt
```

- custom RGB images
- either:
  - a single STL and an image list, or
  - a folder of objects plus instance masks / detections

For grasp visualization, this README assumes:

- you visualize one object at a time
- each object has its own feasible grasp library JSON
- you explicitly pass that JSON to the visualization script
- you do not rely on any hardcoded default grasps

## Repo Files Used in This Workflow

Main entry points:

- `src/scripts/prepare_single_object_img_list_dataset.py`
- `src/scripts/prepare_custom_test_dataset.py`
- `src/scripts/render_templates_from_mesh_dir.py`
- `test_myprints.py`
- `src/scripts/vis_custom_predictions.py`
- `src/scripts/adaptive_map_grasp_point_to_2d.py`

## Output Layout

This custom workflow writes into `gigaPose_datasets/`:

```text
gigaPose_datasets/
  datasets/
    <dataset_name>/
      models/
      test_imagewise/
      test/
      test_targets_bop19.json
    templates/
      <dataset_name>/
        <obj_id>/
        object_poses/
    default_detections/
      core19_model_based_unseen/
        cnos-fastsam/
  results/
    large_<run_id>/
      predictions/
      visualizations/
      adaptive_grasp_point_mapping/
```

## Step 0: Activate Environment

From the repository root:

```bash
conda activate perseve-sam2
```

## Workflow A: One STL + One Image List

Use this when:

- you want to evaluate one object
- you already know which STL you want
- you have a text file containing one RGB path per line

### A1. Prepare the Dataset

```bash
export DATASET=<dataset_name>
export OBJ_ID=<object_id>
export STL_PATH=/path/to/object_mesh.stl
export IMAGE_LIST=/path/to/image_list.txt

python -m src.scripts.prepare_single_object_img_list_dataset \
  --image-list $IMAGE_LIST \
  --mesh-path $STL_PATH \
  --dataset-name $DATASET \
  --dataset-root gigaPose_datasets/datasets \
  --obj-id $OBJ_ID
```

This creates:

- `gigaPose_datasets/datasets/$DATASET/models/obj_<object_id>.obj`
- `gigaPose_datasets/datasets/$DATASET/test_imagewise/...`
- `gigaPose_datasets/datasets/$DATASET/test/shard-000000.tar`
- fallback detections under `default_detections/...`

### A2. Render Templates

```bash
python -m src.scripts.render_templates_from_mesh_dir \
  gigaPose_datasets/datasets/$DATASET/models \
  --output-dir gigaPose_datasets/datasets/templates/$DATASET \
  --renderer blenderproc \
  --num-workers 1 \
  --num-gpus 1 \
  --disable-output \
  --mesh-scale 10
```

Notes:

- `--mesh-scale` is useful for small custom meshes.
- `--renderer panda3d` is available, but BlenderProc is the main path used here.

Sanity check:

- the object-specific template directory should contain both RGBA PNGs and depth PNGs
- rendered template images should have non-empty alpha for at least some views

### A3. Run GigaPose

```bash
python test_myprints.py \
  --dataset-name $DATASET \
  --run-id $DATASET \
  --selected-obj-id $OBJ_ID \
  --test-setting localization
```

Prediction CSV:

```text
gigaPose_datasets/results/large_<DATASET>/predictions/large-pbrreal-rgb-mmodel_<DATASET>-test_<DATASET>.csv
```

### A4. Optional Pose Visualization

```bash
python -m src.scripts.vis_custom_predictions \
  --dataset-dir gigaPose_datasets/datasets/$DATASET \
  --result-csv gigaPose_datasets/results/large_$DATASET/predictions/large-pbrreal-rgb-mmodel_${DATASET}-test_${DATASET}.csv \
  --output-dir gigaPose_datasets/results/large_$DATASET/visualizations
```

### A5. Grasp Visualization for One Object

Prepare one feasible grasp JSON for the object you are visualizing. Then run:

```bash
export FEASIBLE_GRASPS_JSON=/path/to/object_feasible_grasps.json

python -m src.scripts.adaptive_map_grasp_point_to_2d \
  --dataset-dir gigaPose_datasets/datasets/$DATASET \
  --result-csv gigaPose_datasets/results/large_$DATASET/predictions/large-pbrreal-rgb-mmodel_${DATASET}-test_${DATASET}.csv \
  --output-dir gigaPose_datasets/results/large_$DATASET/adaptive_grasp_point_mapping \
  --object-name $OBJ_ID \
  --feasible-grasps-json $FEASIBLE_GRASPS_JSON
```

Important:

- this guide assumes the feasible grasp JSON is the source of grasp candidates
- do not depend on hardcoded default grasps
- visualize one object at a time

If you have real semantic masks for the RGB frames, also pass the mask directory:

```bash
  --mask-annotation-dir /path/to/semantic_mask_folder
```

The script will:

- read the predicted object pose from the GigaPose CSV
- choose the feasible pose that best matches the predicted tabletop orientation
- project the corresponding grasp points to 2D
- optionally recenter the grasp cluster to the object mask centroid

Outputs:

- overlay images
- `projected_points.csv`

## Workflow B: All STLs in a Folder

Use this when:

- you want to onboard all objects in a mesh folder
- your images may contain multiple object classes
- you have instance segmentation masks or equivalent object-level detections

There are two parts:

1. build the dataset with all object meshes and all image annotations
2. render templates for every object in the dataset

### B1. Prepare a Multi-Object Dataset

If your raw dataset already has:

- `rgb_XXXX.png`
- `instance_segmentation_XXXX.png`
- `instance_segmentation_mapping_XXXX.json`

then use:

```bash
export DATASET=<dataset_name>
export RAW_ROOT=/path/to/raw_rgb_and_instance_masks
export MESH_DIR=/path/to/all_meshes

python -m src.scripts.prepare_custom_test_dataset \
  --raw-root $RAW_ROOT \
  --mesh-dir $MESH_DIR \
  --dataset-root gigaPose_datasets/datasets \
  --dataset-name $DATASET
```

This script:

- builds `name_to_obj_id.json` from the mesh folder or `models/`
- writes the dataset layout
- generates object detections from the instance masks
- writes the webdataset shard for GigaPose testing

### B2. Make Sure `models/` Contains `obj_XXXXXX.obj` or `obj_XXXXXX.ply`

`render_templates_from_mesh_dir.py` expects:

```text
gigaPose_datasets/datasets/<dataset_name>/models/obj_000001.obj
gigaPose_datasets/datasets/<dataset_name>/models/obj_000002.obj
...
```

If your meshes start as STL files, convert or symlink them into this naming convention first.

### B3. Render Templates for All Objects

```bash
python -m src.scripts.render_templates_from_mesh_dir \
  gigaPose_datasets/datasets/$DATASET/models \
  --output-dir gigaPose_datasets/datasets/templates/$DATASET \
  --renderer blenderproc \
  --num-workers 1 \
  --num-gpus 1 \
  --disable-output
```

This renders all objects found in `models/`.

### B4. Run GigaPose on the Multi-Object Dataset

```bash
python test_myprints.py \
  --dataset-name $DATASET \
  --run-id $DATASET \
  --test-setting localization
```

If you want to restrict inference to one object:

```bash
python test_myprints.py \
  --dataset-name $DATASET \
  --run-id $DATASET \
  --selected-obj-id 6 \
  --test-setting localization
```

### B5. Visualize Grasps for One Object

Even for a multi-object dataset, grasp visualization is one object at a time:

```bash
python -m src.scripts.adaptive_map_grasp_point_to_2d \
  --dataset-dir gigaPose_datasets/datasets/$DATASET \
  --result-csv gigaPose_datasets/results/large_$DATASET/predictions/large-pbrreal-rgb-mmodel_${DATASET}-test_${DATASET}.csv \
  --output-dir gigaPose_datasets/results/large_$DATASET/adaptive_grasp_point_mapping \
  --object-name <object_id> \
  --feasible-grasps-json /path/to/object_feasible_grasps.json
```
