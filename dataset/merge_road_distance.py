"""
Merge Road_Distance_Export.csv into all_samples_features2_with_id.csv using the `id` column.

Usage (PowerShell):
  python merge_road_distance.py --samples all_samples_features2_with_id.csv --road Road_Distance_Export.csv --out merged.csv

Behavior:
 - Performs a left join: all rows from samples are kept in original order.
 - Adds a new column `dist_to_infra_km` (from road file). If id not found, column left empty.
 - ID matching is tolerant: compares stringified stripped values.
"""
import argparse
import csv
import os


def load_road_dist(path):
    d = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'id' not in (reader.fieldnames or []):
            raise SystemExit(f"Road file {path} missing 'id' column")
        val_col = None
        # find first non-id column to import (dist_to_infra_km expected)
        for fn in reader.fieldnames:
            if fn.lower() != 'id':
                val_col = fn
                break
        if val_col is None:
            raise SystemExit(f"Road file {path} contains only 'id' column and no value column")
        for r in reader:
            key = str(r.get('id','')).strip()
            d[key] = r.get(val_col, '')
        return d, val_col


def merge(samples_path, road_path, out_path):
    road_map, road_val_col = load_road_dist(road_path)
    with open(samples_path, newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if 'id' not in fieldnames:
            raise SystemExit(f"Samples file {samples_path} missing 'id' column")
        # append a dist column name (use existing name from road file)
        out_field = road_val_col
        if out_field in fieldnames:
            # avoid collision
            out_field = road_val_col + '_from_road'
        out_fieldnames = fieldnames + [out_field]

        rows = list(reader)

    # write merged
    with open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows:
            key = str(r.get('id','')).strip()
            val = road_map.get(key, '')
            r_out = {k: r.get(k, '') for k in fieldnames}
            r_out[out_field] = val
            writer.writerow(r_out)

    print(f"Wrote merged file: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', '-s', required=True, help='Path to all_samples_features2_with_id.csv')
    parser.add_argument('--road', '-r', required=True, help='Path to Road_Distance_Export.csv')
    parser.add_argument('--out', '-o', help='Output CSV path (optional)')
    args = parser.parse_args()
    samples = args.samples
    road = args.road
    out = args.out or os.path.splitext(samples)[0] + '_with_road_distance.csv'
    merge(samples, road, out)
