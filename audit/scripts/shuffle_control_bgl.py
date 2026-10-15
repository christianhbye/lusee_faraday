"""Real-beam shuffle control with bounded kernel and NUFFT source batches.

Use --plan first: it inspects sizes/completion without numerical imports or
simulation. Version 2 fixes the negative-branch exponent and effective-count
statistic. It never resumes or overwrites the September legacy ledger.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "audit/generated"
LMAX = 1023
FREQ_MHZ = 30.0
NSIDES = (512, 1024, 2048)
SNAPSHOTS = (0, 682)
FRACTIONS = (1.0, 0.1, 0.02)
PAIR_INDICES = (0, 1)
BRANCH_SIGNS = {"positive": 1, "negative": -1}
RNG_SEED = 20260818
SCHEMA_VERSION = 2
RESULT_PATH = OUT / "shuffle_bgl_v2_results.json"
CACHE_PATH = ROOT / "generated_data/cache/shuffle_bgl_v2"
SKY_CACHE = ROOT / "generated_data/cache/sky_maps_native.npz"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsides", type=int, nargs="+", default=NSIDES)
    parser.add_argument("--snapshots", type=int, nargs="+", default=SNAPSHOTS)
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--nufft-chunk", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_PATH)
    parser.add_argument("--save-spectra", action="store_true")
    parser.add_argument("--plan", action="store_true", help="no simulation")
    args = parser.parse_args(argv)
    if any(n not in NSIDES for n in args.nsides):
        parser.error(f"--nsides must be drawn from {NSIDES}")
    if any(t < 0 or t >= 1024 for t in args.snapshots):
        parser.error("--snapshots must be in [0, 1024)")
    if min(args.chunk, args.nufft_chunk, args.threads) <= 0:
        parser.error("batch sizes and thread count must be positive")
    args.nsides = sorted(set(args.nsides))
    args.snapshots = sorted(set(args.snapshots))
    args.result_path = args.result_path.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.result_path == (OUT / "shuffle_bgl_results.json").resolve():
        parser.error("preserve the September legacy ledger")
    return args


def load_numerics(threads=2):
    """Import numerical dependencies only for calculations or tests."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    global np, hp, finufft, rsp, cfg, conv
    import numpy as np
    import healpy as hp
    import finufft
    from lusee_faraday import response as rsp
    from lusee_faraday import config as cfg
    from lusee_faraday import conventions as conv


def run_key(nside, snapshot, pair, branch, fraction, mask):
    return f"n{nside}_t{snapshot}_p{pair}_{branch}_f{fraction:g}_{mask}"


def expected_keys(args):
    return [
        run_key(ns, t, pair, branch, fraction, mask)
        for ns in args.nsides
        for t in args.snapshots
        for pair in PAIR_INDICES
        for branch in BRANCH_SIGNS
        for fraction in FRACTIONS
        for mask in ("horizon", "theta86")
    ]


def preflight(args):
    """Read-only inspection; array sizes are not a peak-RSS guarantee."""
    existing = {}
    if args.result_path.exists():
        existing = json.loads(args.result_path.read_text())
        if (
            existing.get("metadata", {}).get("schema_version")
            != SCHEMA_VERSION
        ):
            raise RuntimeError("incompatible ledger; choose a new result path")
    keys = expected_keys(args)
    completed = sum(k in existing.get("runs", {}) for k in keys)
    print(f"ledger: {args.result_path}")
    print(f"requested entries: {len(keys)}; present: {completed}")
    print("Execution also validates the input/code/environment fingerprint.")
    print(f"kernel batch {args.chunk}; NUFFT sources {args.nufft_chunk}")
    print(f"threads {args.threads}; scratch {args.cache_dir}")
    print(f"native sky cache present: {SKY_CACHE.exists()}")
    for ns in args.nsides:
        npix = 12 * ns**2
        resident = npix * (3 * 8 + 8)  # RM, Q, U and permutation
        scratch = npix // 2 * (8 + 8 + 4 * 16)
        sample = args.chunk * 10 * 4 * 16
        print(
            f"nside {ns}: maps+permutation {resident / 2**30:.2f} GiB; "
            f"scratch/snapshot {scratch / 2**30:.2f} GiB; "
            f"one sampled K array <= {sample / 2**20:.1f} MiB"
        )
    print(
        "Add RAM for selected RM arrays, interpolation, transforms, FINUFFT."
    )
    print("Scratch is rebuilt on restart; ledger observables are preserved.")
    print("No calculation launched; no files written.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(directory=ROOT):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def fingerprint():
    """Hash effective input bytes, numerical code and editable dependencies."""
    from urllib.parse import unquote, urlparse

    if not SKY_CACHE.exists():
        raise RuntimeError(
            "native sky cache missing; prepare it before running"
        )
    response_path = Path(
        os.environ.get(
            "LUSEE_RESPONSE",
            str(ROOT / "data/BGL_v16/lusee_bgl_v16_response_v3.fits"),
        )
    ).resolve()
    sources = [Path(__file__), ROOT / "scripts/common.py"]
    sources.extend(sorted((ROOT / "src/lusee_faraday").glob("*.py")))
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "healpy",
        "finufft",
        "lusee",
        "croissant-sim",
        "astropy",
        "lunarsky",
        "h5py",
        "fitsio",
        "jax",
        "jaxlib",
    ):
        dist = importlib.metadata.distribution(name)
        entry = {"version": dist.version}
        direct = dist.read_text("direct_url.json")
        if direct:
            entry["direct_url"] = json.loads(direct)
            parsed = urlparse(entry["direct_url"].get("url", ""))
            if parsed.scheme == "file":
                checkout = Path(unquote(parsed.path))
                entry["git_revision"] = git_revision(checkout)
                diff = subprocess.check_output(
                    ["git", "diff", "HEAD", "--", "*.py", "pyproject.toml"],
                    cwd=checkout,
                )
                entry["diff_sha256"] = hashlib.sha256(diff).hexdigest()
        packages[name] = entry
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs_sha256": {
            str(p): sha256(p) for p in (response_path, SKY_CACHE)
        },
        "sources_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in sources
        },
        "packages": packages,
        "python": sys.version,
        "frequency_mhz": FREQ_MHZ,
        "lmax": LMAX,
        "fractions": list(FRACTIONS),
        "pair_indices": list(PAIR_INDICES),
        "rng_seed": RNG_SEED,
        "branch_signs": BRANCH_SIGNS,
        "n_eff_definition": "sum(abs(w))**2 / sum(abs(w)**2)",
        "sky_cache_policy": "effective native cache bytes hashed",
        "response_path": str(response_path),
    }
    token = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    metadata["fingerprint"] = token.hexdigest()
    return metadata


@contextmanager
def ledger_lock(path):
    """Refuse concurrent writers to one result ledger."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another job owns {path}") from error
        yield


def write_results(results, path):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def load_results(args, metadata):
    if args.result_path.exists():
        results = json.loads(args.result_path.read_text())
        if (
            results.get("metadata", {}).get("fingerprint")
            != metadata["fingerprint"]
        ):
            raise RuntimeError(
                "incompatible provenance; choose a new --result-path, "
                "preserving the old ledger"
            )
        return results
    metadata = dict(metadata)
    metadata.update(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_revision=git_revision(),
    )
    return {"metadata": metadata, "runs": {}, "executions": []}


def memory_note(stage):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"[{stage}] peak RSS so far {peak:.2f} GiB", flush=True)


def sky_alms():
    """Hold the same band-limited fields fixed across quadrature refinement."""
    from scripts.common import load_sky_maps

    maps = load_sky_maps()
    scale = (FREQ_MHZ / cfg.FREQ_REF_QU) ** cfg.BETA_QU
    return tuple(
        hp.map2alm(np.asarray(m, dtype=np.float64), lmax=LMAX, iter=0)
        for m in (maps["RM"], maps["Q23"] * scale, maps["U23"] * scale)
    )


def target_maps(alms, nside):
    return tuple(hp.alm2map(alm, nside, lmax=LMAX) for alm in alms)


def direction_chunk(nside, pixels, rotation):
    """Response angles and tangent-basis transport, bounded by pixel batch."""
    theta_gal, phi_gal = hp.pix2ang(nside, pixels)
    st, ct = np.sin(theta_gal), np.cos(theta_gal)
    sp, cp = np.sin(phi_gal), np.cos(phi_gal)
    vec = np.stack((st * cp, st * sp, ct), axis=-1)
    e_theta = np.stack((ct * cp, ct * sp, -st), axis=-1)
    vec_topo = vec @ rotation.T
    theta = np.arccos(np.clip(vec_topo[:, 2], -1.0, 1.0))
    phi = np.arctan2(vec_topo[:, 1], vec_topo[:, 0]) % (2 * np.pi)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e_phi_topo = np.stack((-sp, cp, np.zeros_like(sp)), axis=-1)
    e_theta_topo = np.stack((ct * cp, ct * sp, -st), axis=-1)
    transported = e_theta @ rotation.T
    alpha = np.arctan2(
        np.sum(transported * e_phi_topo, axis=-1),
        np.sum(transported * e_theta_topo, axis=-1),
    )
    return theta, phi, alpha


def branch_weights(kq, ku, p):
    """COSMO amplitudes; complex cross-pair branches are independent."""
    return {
        "positive": 0.5 * (kq - 1j * ku) * p,
        "negative": 0.5 * (kq + 1j * ku) * np.conj(p),
    }


def pair_weights(kernel, nside, rotation, q_map, u_map, chunk, directory):
    """Disk-backed weights avoid keeping four full complex arrays in RAM."""
    npix = hp.nside2npix(nside)
    nup = npix // 2  # antipodal centres, generically rotated horizon
    directory.mkdir(parents=True, exist_ok=True)

    def allocate(name, dtype):
        return np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype, shape=(nup,)
        )

    selection = allocate("selection", np.int64)
    theta_up = allocate("theta", np.float64)
    out = {
        pair: {
            branch: allocate(f"p{pair}_{branch}", np.complex128)
            for branch in BRANCH_SIGNS
        }
        for pair in PAIR_INDICES
    }
    filled = 0
    for start in range(0, npix, chunk):
        stop = min(start + chunk, npix)
        pixels = np.arange(start, stop, dtype=np.int64)
        theta, phi, alpha = direction_chunk(nside, pixels, rotation)
        keep = theta <= np.pi / 2 + 1e-12
        pixels = pixels[keep]
        count = pixels.size
        end = filled + count
        if not count:
            continue
        if end > nup:
            raise RuntimeError("above-horizon count exceeded allocation")
        selection[filled:end] = pixels
        theta_up[filled:end] = theta[keep]
        q_iau, u_iau = conv.cosmo_to_iau_qu(q_map[pixels], u_map[pixels])
        p = (q_iau - 1j * u_iau) * np.exp(2j * alpha[keep])
        sampled = kernel.sample(theta[keep], phi[keep])
        for pair in PAIR_INDICES:
            amplitudes = branch_weights(sampled[pair, 1], sampled[pair, 2], p)
            for branch, values in amplitudes.items():
                out[pair][branch][filled:end] = values
        filled = end
        del sampled, amplitudes, values
        if stop % (4 * chunk) == 0 or stop == npix:
            print(f"    pixels {stop}/{npix}; samples {filled}", flush=True)
    if filled != nup:
        raise RuntimeError(f"expected {nup} above horizon, found {filled}")
    for array in (selection, theta_up):
        array.flush()
    for pair in out.values():
        for weight in pair.values():
            weight.flush()
    memory_note("kernel sampled into disk-backed arrays")
    return selection, theta_up, out


def visibility(rm, weight, fraction, lam2, sign, chunk, threads=2, mask=None):
    """Sum complex source-batch visibilities BEFORE forming power.

    Selecting a mask within each batch avoids full-size masked copies.
    Linearity makes this the same estimator as one full type-3 NUFFT.
    """
    if sign not in (-1, 1):
        raise ValueError("Faraday branch sign must be +1 or -1")
    if chunk <= 0:
        raise ValueError("NUFFT batch size must be positive")
    vis = np.zeros(lam2.size, dtype=np.complex128)
    for start in range(0, rm.size, chunk):
        end = min(start + chunk, rm.size)
        r, w = rm[start:end], weight[start:end]
        if mask is not None:
            keep = mask[start:end]
            r, w = r[keep], w[keep]
        if r.size:
            vis += finufft.nufft1d3(
                np.ascontiguousarray(sign * 2.0 * fraction * r),
                np.ascontiguousarray(w),
                np.ascontiguousarray(lam2),
                eps=1e-9,
                isign=1,
                nthreads=threads,
            )
    return vis


def power_spectrum(rm, weight, fraction, lam2, hann, **kwargs):
    vis = visibility(rm, weight, fraction, lam2, **kwargs)
    return np.abs(np.fft.fftshift(np.fft.fft(vis * hann))) ** 2


def weight_statistics(weight, chunk, mask=None):
    l1, l2, coherent, count = 0.0, 0.0, 0j, 0
    for start in range(0, weight.size, chunk):
        end = start + chunk
        w = weight[start:end]
        if mask is not None:
            w = w[mask[start:end]]
        amplitude = np.abs(w)
        l1 += float(amplitude.sum())
        l2 += float(np.dot(amplitude, amplitude))
        coherent += w.sum()
        count += w.size
    if not count or not l2:
        raise RuntimeError("empty or zero-weight observable")
    return {
        "pixels": count,
        "weight_power": l2,
        "n_eff_amplitude": l1**2 / l2,
        "unrotated_coherence": float(abs(coherent) / l1),
    }


def spectrum_path(args, key):
    return args.result_path.parent / f"{args.result_path.stem}_{key}.npz"


def is_complete(args, results, key):
    if key not in results["runs"]:
        return False
    save = args.save_spectra and key.endswith("_horizon")
    return not save or spectrum_path(args, key).exists()


def calculate(args):
    results = load_results(args, fingerprint())
    execution = dict(
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        nsides=args.nsides,
        snapshots=args.snapshots,
        chunk=args.chunk,
        nufft_chunk=args.nufft_chunk,
        threads=args.threads,
    )
    results["executions"].append(execution)
    write_results(results, args.result_path)
    wanted = expected_keys(args)
    if all(is_complete(args, results, k) for k in wanted):
        execution["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        execution["skipped_all"] = True
        write_results(results, args.result_path)
        print("All requested observables present; no maps loaded.", flush=True)
        return
    alms = sky_alms()
    response = rsp.load_response(results["metadata"]["response_path"])
    kernel = rsp.FixedChannelKernel(response, FREQ_MHZ)
    del response
    gc.collect()
    memory_note("native inputs and frozen response loaded")
    lam2 = conv.lambda_squared(cfg.fine_freqs(FREQ_MHZ))
    hann = np.hanning(lam2.size)
    h2_sum = float(np.sum(hann**2))
    obs_times, loc = cfg.times(), cfg.moon_location()
    # Separate ledgers also get separate scratch directories.
    ledger_id = hashlib.sha256(str(args.result_path).encode()).hexdigest()[:12]
    scratch = args.cache_dir / results["metadata"]["fingerprint"] / ledger_id
    print(f"results/resume ledger: {args.result_path}", flush=True)
    for ns in args.nsides:
        ns_keys = [k for k in wanted if k.startswith(f"n{ns}_")]
        if all(is_complete(args, results, k) for k in ns_keys):
            continue
        rm, q_map, u_map = target_maps(alms, ns)
        pixel_area = 4 * np.pi / hp.nside2npix(ns)
        permutation = np.random.default_rng(
            np.random.SeedSequence([RNG_SEED, ns])
        ).permutation(rm.size)
        memory_note(f"nside {ns} maps and permutation")
        for snapshot in args.snapshots:
            keys = [k for k in ns_keys if k.startswith(f"n{ns}_t{snapshot}_")]
            if all(is_complete(args, results, k) for k in keys):
                continue
            rotation = conv.topo_rotation_matrix(obs_times[snapshot], loc)
            # Scratch is rebuilt: an interrupted partial file is never reused.
            selection, theta_up, weights = pair_weights(
                kernel,
                ns,
                rotation,
                q_map,
                u_map,
                args.chunk,
                scratch / f"n{ns}_t{snapshot}",
            )
            rm_up = rm[selection]
            shuffled_up = rm[permutation[selection]]
            mask86 = theta_up <= np.radians(86.0)
            for pair in PAIR_INDICES:
                for branch, weight in weights[pair].items():
                    weight *= pixel_area
                    for mask_name, mask in (
                        ("horizon", None),
                        ("theta86", mask86),
                    ):
                        stats = weight_statistics(
                            weight, args.nufft_chunk, mask
                        )
                        incoh = lam2.size * h2_sum * stats["weight_power"]
                        for fraction in FRACTIONS:
                            key = run_key(
                                ns, snapshot, pair, branch, fraction, mask_name
                            )
                            if is_complete(args, results, key):
                                continue
                            kwargs = dict(
                                sign=BRANCH_SIGNS[branch],
                                chunk=args.nufft_chunk,
                                threads=args.threads,
                                mask=mask,
                            )
                            real = power_spectrum(
                                rm_up, weight, fraction, lam2, hann, **kwargs
                            )
                            shuffled = power_spectrum(
                                shuffled_up,
                                weight,
                                fraction,
                                lam2,
                                hann,
                                **kwargs,
                            )
                            entry = dict(
                                nside=ns,
                                snapshot=snapshot,
                                pair_index=pair,
                                pair=list(conv.PORT_PAIRS[pair]),
                                branch=branch,
                                phase_sign=BRANCH_SIGNS[branch],
                                fraction=fraction,
                                mask=mask_name,
                                pixel_area_sr=pixel_area,
                                **stats,
                                real=float(real.sum()),
                                shuffled=float(shuffled.sum()),
                                incoherent=incoh,
                            )
                            entry["real_over_shuffled"] = (
                                entry["real"] / entry["shuffled"]
                            )
                            entry["real_over_incoherent"] = (
                                entry["real"] / incoh
                            )
                            if args.save_spectra and mask_name == "horizon":
                                path = spectrum_path(args, key)
                                temp = path.with_suffix(".tmp.npz")
                                np.savez_compressed(
                                    temp,
                                    lambda_squared=lam2,
                                    real=real,
                                    shuffled=shuffled,
                                )
                                temp.replace(path)
                            results["runs"][key] = entry
                            write_results(results, args.result_path)
                            print(
                                f"{key}: "
                                f"real/shuf={entry['real_over_shuffled']:.3g} "
                                "real/incoh="
                                f"{entry['real_over_incoherent']:.3g} "
                                f"N_eff={stats['n_eff_amplitude']:.5g}",
                                flush=True,
                            )
                            memory_note(key)
            del (
                selection,
                theta_up,
                weights,
                weight,
                mask86,
                mask,
                rm_up,
                shuffled_up,
            )
            gc.collect()
        del rm, q_map, u_map, permutation
        gc.collect()
    execution["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    write_results(results, args.result_path)
    print("Requested observables complete.", flush=True)
    memory_note("finished")


def main():
    args = arguments()
    if args.plan:
        preflight(args)
        return
    load_numerics(args.threads)
    with ledger_lock(args.result_path):
        calculate(args)


if __name__ == "__main__":
    main()
