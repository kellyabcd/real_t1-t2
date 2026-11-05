"""Register numbered IXI T1 volumes to each subject's native T2 grid.

Requires: pip install SimpleITK numpy
Run a small pilot and inspect overlays before processing the whole dataset.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VERSION = "physical_identity_qc_v2"


def normalised_mutual_information(a, b, mask):
    """Fixed support/range for every candidate; zeros remain in the histogram."""
    histogram = np.histogram2d(
        a[mask], b[mask], bins=64, range=((0.0, 1.0), (0.0, 1.0))
    )[0]
    probability = histogram / max(histogram.sum(), 1.0)
    def entropy(values):
        values = values[values > 0]
        return -float(np.sum(values * np.log(values)))
    joint = entropy(probability)
    return (entropy(probability.sum(0)) + entropy(probability.sum(1))) / max(joint, 1e-12)


def select_result(baseline, candidate, fixed, moving, fixed_mask):
    # Independent deterministic QC on a fixed lattice, distinct from random
    # Mattes MI samples used by the optimiser. Masks never depend on candidate.
    target = sitk.GetArrayViewFromImage(fixed)[::2, ::2, ::2]
    mask = sitk.GetArrayFromImage(fixed_mask)[::2, ::2, ::2] > 0
    source_values = sitk.GetArrayViewFromImage(moving)[::2, ::2, ::2]
    source_positive = source_values[source_values > 0]
    target_positive = target[target > 0]
    if mask.sum() < 1000 or len(source_positive) == 0 or len(target_positive) == 0:
        raise ValueError("Insufficient foreground for reliable quality control")
    source_scale = max(float(np.percentile(source_positive, 99.5)), 1e-8)
    target_scale = max(float(np.percentile(target_positive, 99.5)), 1e-8)
    target_norm = np.clip(target / target_scale, 0, 1)
    scores = {}
    for key, image in (("baseline", baseline), ("candidate", candidate)):
        values = sitk.GetArrayViewFromImage(image)[::2, ::2, ::2]
        scores[key] = {
            "fixed_mask_nmi": normalised_mutual_information(
                np.clip(values / source_scale, 0, 1), target_norm, mask
            ),
            "nonzero_coverage": float(np.mean(values[mask] > 0)),
        }
    better = scores["candidate"]["fixed_mask_nmi"] > scores["baseline"]["fixed_mask_nmi"] + 1e-5
    coverage_ok = scores["candidate"]["nonzero_coverage"] >= scores["baseline"]["nonzero_coverage"] - 0.005
    accepted = bool(better and coverage_ok)
    return accepted, {
        **scores, "accepted_candidate": accepted,
        "decision": "optimised_rigid" if accepted else "physical_identity_fallback",
        "reason": "NMI improved and coverage retained" if accepted else "Candidate did not pass NMI/coverage safeguards",
        "definition": "64-bin NMI; T2 Otsu+dilation fixed mask; step-2 lattice; original-volume 99.5th-percentile ranges; candidate zeros included",
    }


def index_files(folder, prefix):
    pattern = re.compile(rf"^{prefix}_(\d+)\.nii(?:\.gz)?$")
    original_pattern = re.compile(rf"^IXI(\d+)-[^-]+-[^-]+-{prefix}\.nii(?:\.gz)?$")
    indexed = {}
    for path in Path(folder).iterdir():
        match = pattern.fullmatch(path.name) or original_pattern.fullmatch(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if number in indexed:
            raise ValueError(f"Duplicate numbered volume: {indexed[number]} and {path}")
        indexed[number] = path
    return indexed


def original_pair_key(path):
    match = re.fullmatch(r"(IXI\d+-[^-]+-[^-]+)-(T1|T2)\.nii(?:\.gz)?", path.name)
    return match.group(1) if match else None


def register(t1_path, t2_path, dest, compact=False):
    fixed = sitk.ReadImage(str(t2_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(t1_path), sitk.sitkFloat32)
    fixed_mask = sitk.BinaryDilate(sitk.OtsuThreshold(fixed, 0, 1, 128), [3, 3, 1])
    initial = sitk.Euler3DTransform()
    initial.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2 for size in fixed.GetSize()]
    ))
    # Center is only the rotation anchor. Translation remains zero, preserving
    # scanner physical alignment instead of aligning unequal volume centres.
    baseline = sitk.Resample(moving, fixed, initial, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(50)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, 2026)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5, minStep=0.0001, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    failure = None
    try:
        candidate_transform = method.Execute(fixed, moving)
        candidate = sitk.Resample(moving, fixed, candidate_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        accepted, qc = select_result(baseline, candidate, fixed, moving, fixed_mask)
        sitk.WriteTransform(candidate_transform, str(dest / "candidate_T1_to_T2.tfm"))
    except RuntimeError as exc:
        failure = str(exc)
        accepted = False
        qc = {"accepted_candidate": False, "decision": "physical_identity_fallback", "reason": "Registration optimiser failed", "error": failure}
    transform = candidate_transform if accepted else initial
    aligned = candidate if accepted else baseline
    if (aligned.GetSize() != fixed.GetSize()
            or not np.allclose(aligned.GetDirection(), fixed.GetDirection())
            or not np.allclose(aligned.GetOrigin(), fixed.GetOrigin())
            or not np.allclose(aligned.GetSpacing(), fixed.GetSpacing())):
        raise RuntimeError("Output grid does not match T2")
    if not compact:
        sitk.WriteImage(baseline, str(dest / "T1_physical_baseline.nii.gz"))
    sitk.WriteImage(aligned, str(dest / "T1_registered.nii.gz"))
    sitk.WriteTransform(transform, str(dest / "T1_to_T2.tfm"))
    # This normalized image is for overlay inspection only, never training.
    if not compact:
        a = sitk.GetArrayFromImage(aligned)
        b = sitk.GetArrayFromImage(fixed)
        norm = lambda x: (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
        overlay = sitk.GetImageFromArray(np.stack([norm(a), norm(b), norm(a)], axis=-1).astype(np.float32), isVector=True)
        overlay.CopyInformation(fixed)
        sitk.WriteImage(overlay, str(dest / "qc_T1_magenta_T2_green.nii.gz"))
    return {
        "t1": str(t1_path), "t2": str(t2_path),
        "output": str(dest / "T1_registered.nii.gz"),
        "size_xyz": list(fixed.GetSize()), "spacing_xyz_mm": list(fixed.GetSpacing()),
        "metric": method.GetMetricValue() if failure is None else None,
        "optimizer_stop": method.GetOptimizerStopConditionDescription(),
        "registration": "rigid 3D MI; identity physical initialisation; independent QC fallback; T2 fixed",
        "version": VERSION,
        "quality_control": qc,
        "qc_status": "manual review required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1_dir", required=True)
    parser.add_argument("--t2_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0; zero processes all paired volumes")
    t1, t2 = index_files(args.t1_dir, "T1"), index_files(args.t2_dir, "T2")
    numbers = sorted(t1.keys() & t2.keys())
    for number in numbers:
        a, b = original_pair_key(t1[number]), original_pair_key(t2[number])
        if a != b:
            raise ValueError(f"Subject/site/scan mismatch: {t1[number]} versus {t2[number]}")
    if args.limit:
        numbers = numbers[:args.limit]
    output = Path(args.output_dir).resolve()
    if output in (Path(args.t1_dir).resolve(), Path(args.t2_dir).resolve()):
        raise ValueError("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Matched pairs: {len(t1.keys() & t2.keys())}; selected: {len(numbers)}")
    original_mode = bool(numbers) and original_pair_key(t1[numbers[0]]) is not None
    print("Pairing verified by original subject/site/scan names." if original_mode
          else "Numeric pairing assumes the existing ID-matched renaming was preserved.")
    with (output / "paired_subjects.csv").open("w", encoding="utf-8-sig", newline="") as pair_handle:
        pair_writer = csv.writer(pair_handle)
        pair_writer.writerow(["subject", "t1", "t2"])
        for number in numbers:
            pair_writer.writerow([original_pair_key(t1[number]) or f"pair_{number:04d}", t1[number], t2[number]])
    with (output / "registration_log.csv").open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "status", "detail"])
        for number in numbers:
            subject = original_pair_key(t1[number]) or f"pair_{number:04d}"
            dest = output / subject
            dest.mkdir(exist_ok=True)
            if (dest / "registration.json").exists():
                old = json.loads((dest / "registration.json").read_text(encoding="utf-8"))
                if old.get("version") != VERSION:
                    raise ValueError("Existing outputs are from another version; select a new output directory")
                print(number, "already processed with current version; skipped")
                continue
            try:
                metadata = register(t1[number], t2[number], dest)
                metadata["subject"] = subject
                (dest / "registration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                writer.writerow([number, "completed_needs_QC", metadata["metric"]])
                print(number, metadata["quality_control"], flush=True)
            except Exception as exc:
                writer.writerow([number, "failed", str(exc)])
                print(number, "FAILED", str(exc))
            handle.flush()


if __name__ == "__main__":
    main()
