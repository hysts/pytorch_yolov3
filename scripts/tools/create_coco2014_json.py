#!/usr/bin/env python

import argparse
import json
import os
import pathlib
import tqdm


def load_val_ids(path):
    with open(path, 'r') as fin:
        paths = [line.strip() for line in fin.readlines()]
    ids = [int(line.split('_')[-1].split('.')[0]) for line in paths]
    return ids


def split_annotations(annotation_dir, val_ids):
    with open(annotation_dir / 'instances_train2017.json', 'r') as fin:
        train_data = json.load(fin)
    with open(annotation_dir / 'instances_val2017.json', 'r') as fin:
        val_data = json.load(fin)
    annotations = train_data['annotations'] + val_data['annotations']
    images = train_data['images'] + val_data['images']

    train_annotations = []
    val_annotations = []
    for annotation in tqdm.tqdm(annotations):
        if annotation['image_id'] in val_ids:
            val_annotations.append(annotation)
        else:
            train_annotations.append(annotation)

    train_images = []
    val_images = []
    for image in tqdm.tqdm(images):
        if image['id'] in val_ids:
            val_images.append(image)
        else:
            train_images.append(image)

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': train_data['categories'],
    }
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': val_data['categories'],
    }
    return train_data, val_data


def split_image_paths(coco2017_dir, val_ids):
    train_paths = sorted((coco2017_dir / 'train2017').glob('*'))
    val_paths = sorted((coco2017_dir / 'val2017').glob('*'))
    paths = train_paths + val_paths

    train_paths = []
    val_paths = []
    for path in tqdm.tqdm(paths):
        image_id = int(path.name.split('.')[0])
        if image_id in val_ids:
            val_paths.append(path)
        else:
            train_paths.append(path)
    return train_paths, val_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--coco2017_dir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    val_list_path = pathlib.Path(args.val_list)
    val_ids = load_val_ids(val_list_path)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    coco2017_dir = pathlib.Path(args.coco2017_dir)
    train_data, val_data = split_annotations(coco2017_dir / 'annotations',
                                             val_ids)
    train_paths, val_paths = split_image_paths(coco2017_dir, val_ids)

    annotation_dir = outdir / 'annotations'
    annotation_dir.mkdir(exist_ok=True, parents=True)
    with open(annotation_dir / 'instances_train2014.json', 'w') as fout:
        json.dump(train_data, fout)
    with open(annotation_dir / 'instances_val2014.json', 'w') as fout:
        json.dump(val_data, fout)

    train_image_dir = outdir / 'train2014'
    val_image_dir = outdir / 'val2014'
    train_image_dir.mkdir(exist_ok=True, parents=True)
    val_image_dir.mkdir(exist_ok=True, parents=True)
    for path in train_paths:
        os.symlink(path, train_image_dir / path.name)
    for path in val_paths:
        os.symlink(path, val_image_dir / path.name)


if __name__ == '__main__':
    main()
