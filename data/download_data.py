#!/usr/bin/env python3
"""Prepare MotifGate inputs from the raw JASPAR and UniBind downloads.

MotifGate expects this layout under --out (default ./Shared):

    <out>/transfac/        one TRANSFAC file per motif  (PWM + CC tf_family/tf_class)
    <out>/Jaspar_pwms/     one *.jaspar file per motif  (PWM, optional/redundant)
    <out>/sites/           one <MOTIF_ID>.sites per motif (positive TFBS sequences)

The file name is the join key: load_raw_positive_examples_min_len() derives the
motif_id from the .sites file name and looks up the PWM and tf_family/tf_class by
that id. Version mismatches are fine: a sites file MA0001.1.sites is matched to a
PWM MA0001.3.transfac via the version-stripped base id (MA0001).

------------------------------------------------------------------------------
RAW DOWNLOADS (manual, from the official sites)
------------------------------------------------------------------------------
1) JASPAR 2026 — "JASPAR collections PFMs (non-redundant)"  https://jaspar.elixir.no/downloads/
   You need TWO things from JASPAR:
     (a) the PFMs in TRANSFAC format (per-motif .transfac files, OR one combined
         file split by `//`). TRANSFAC carries the CC tf_family/tf_class annotation.
     (b) the binding-site sequences: JASPAR ships these as per-motif <MA_id>.sites
         files (e.g. MA0001.1.sites) — already in the exact format MotifGate needs.

2) UniBind (external validation) — "FASTA files for the TFBSs per TF"  https://unibind.uio.no/downloads/
   https://unibind.uio.no/static/data/20220914/bulk_Robust/All/damo_fasta_per_TF.tar
   Files are named by TF only (e.g. RUNX1.fasta). UniBind does not cover every
   JASPAR family/class, so some TFs will be unmapped and skipped (logged).

Record the JASPAR release and the access date in REPRODUCIBILITY.md.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
# JASPAR — point at the folders/files you downloaded (split happens only if you
# pass a combined file; per-motif folders are copied as-is):
python data/download_data.py jaspar \
    --transfac-dir jaspar_transfac/ \
    --sites-dir   jaspar_sites/ \
    --out ./Shared
# (or, if you have a single combined TRANSFAC file:)
python data/download_data.py jaspar --transfac JASPAR2026_CORE_nr_pfms_transfac.txt \
    --sites-dir jaspar_sites/ --out ./Shared

# UniBind external-validation set (TF-named FASTA -> <MA>.sites, K in [7,14]):
python data/download_data.py unibind --tar damo_fasta_per_TF.tar \
    --jaspar ./Shared --out ./Shared_unibind
"""
from __future__ import annotations
import argparse, os, re, glob, shutil, tarfile


# --------------------------- JASPAR PFMs (transfac) ---------------------------
def split_combined_transfac(path: str, tdir: str) -> int:
    os.makedirs(tdir, exist_ok=True)
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        block, ac = [], None
        for line in f:
            block.append(line); s = line.strip()
            if s.startswith("AC"):
                p = s.split()
                if len(p) >= 2:
                    ac = p[1]
            if s.startswith("//"):
                if ac:
                    open(os.path.join(tdir, f"{ac}.transfac"), "w").write("".join(block)); n += 1
                block, ac = [], None
    return n


def split_combined_jaspar(path: str, jdir: str) -> int:
    os.makedirs(jdir, exist_ok=True)
    n = 0
    rec, mid = [], None
    def flush():
        nonlocal n
        if mid and rec:
            open(os.path.join(jdir, f"{mid}.jaspar"), "w").write("".join(rec)); n += 1
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                flush(); rec, mid = [line], line[1:].split()[0]
            else:
                rec.append(line)
    flush()
    return n


def copy_dir(src: str, dst: str, patterns) -> int:
    os.makedirs(dst, exist_ok=True)
    seen = set()
    for pat in patterns:
        for fp in glob.glob(os.path.join(src, pat)):
            b = os.path.basename(fp)
            if b in seen:
                continue
            shutil.copy2(fp, os.path.join(dst, b)); seen.add(b)
    return len(seen)


# ------------------------------ annotations ----------------------------------
def load_annotations(shared_dir: str) -> dict:
    """{motif_id: {'tf_family','tf_class','names': set()}} from transfac/ files."""
    ann = {}
    for fp in glob.glob(os.path.join(shared_dir, "transfac", "*")):
        text = open(fp, "r", encoding="utf-8", errors="ignore").read()
        ac = re.search(r"(?:^|\n)AC\s+(\S+)", text)
        if not ac:
            continue
        fam = re.search(r"CC\s+tf_family\s*:\s*(.+)", text)
        cls = re.search(r"CC\s+tf_class\s*:\s*(.+)", text)
        names = set(re.findall(r"(?:^|\n)ID\s+(.+)", text))
        de = re.search(r"(?:^|\n)DE\s+\S+\s+(\S+)", text)
        if de:
            names.add(de.group(1))
        ann[ac.group(1)] = {
            "tf_family": fam.group(1).strip() if fam else "",
            "tf_class": cls.group(1).strip() if cls else "",
            "names": {n.strip().upper() for n in names if n.strip()},
        }
    return ann


def build_name_map(ann: dict) -> dict:
    """TF-name(upper) -> [motif_ids], latest version first. Dimer-aware: each
    component of names like AHR::ARNT is indexed separately so a UniBind file
    RUNX1.fasta maps to RUNX1::* matrices too."""
    m = {}
    for mid, a in ann.items():
        comps = set()
        for nm in a["names"]:
            comps.add(nm)
            for part in re.split(r"::|/|\+", nm):
                part = part.strip()
                if part:
                    comps.add(part)
        for c in comps:
            m.setdefault(c, []).append(mid)
    def ver(x):
        try: return float(x.split(".")[1])
        except Exception: return 0.0
    for k in m:
        m[k] = sorted(set(m[k]), key=ver, reverse=True)
    return m


# ------------------------------ sites / FASTA --------------------------------
def read_fasta(path: str):
    seqs, cur = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    if cur: seqs.append("".join(cur))
    return [s.upper() for s in seqs if s]


def unibind_to_sites(fasta_dir: str, out: str, ann: dict, kmin: int, kmax: int,
                     all_versions: bool=False) -> None:
    sdir = os.path.join(out, "sites")
    os.makedirs(sdir, exist_ok=True)
    name_map = build_name_map(ann)
    files = []
    for pat in ("*.fa", "*.fasta", "*.fas", "*.txt"):
        files += glob.glob(os.path.join(fasta_dir, pat))
    matched, written, kept, unmatched = 0, 0, 0, []
    for fp in sorted(files):
        stem = os.path.basename(fp)
        ma = re.search(r"(MA\d+\.\d+)", stem)
        if ma:
            targets = [ma.group(1)]
        else:
            tf = re.split(r"[._]", stem)[0].upper()
            targets = name_map.get(tf, [])
            if targets and not all_versions:
                targets = targets[:1]
        if not targets:
            unmatched.append(stem); continue
        seqs = [s for s in read_fasta(fp) if kmin <= len(s) <= kmax and set(s) <= set("ACGT")]
        if not seqs:
            continue
        for mid in targets:
            with open(os.path.join(sdir, f"{mid}.sites"), "a") as g:
                g.write("\n".join(seqs) + "\n")
            written += 1; kept += len(seqs)
        matched += 1
    print(f"[unibind] mapped {matched} TF files -> {written} .sites files "
          f"({kept} sequences in length [{kmin},{kmax}]) in {sdir}")
    if unmatched:
        print(f"[unibind] {len(unmatched)} TF files unmapped (not in JASPAR family/class set); "
              f"first few: {unmatched[:5]}")
        open(os.path.join(out, "unmatched_unibind.txt"), "w").write("\n".join(unmatched) + "\n")


# ----------------------------------- CLI -------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MotifGate inputs (JASPAR + UniBind).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("jaspar", help="assemble Shared/ from JASPAR PFMs + sites")
    j.add_argument("--transfac", help="combined TRANSFAC file (will be split by //)")
    j.add_argument("--transfac-dir", help="folder of per-motif .transfac files (copied as-is)")
    j.add_argument("--jaspar", help="combined .jaspar file (optional; will be split)")
    j.add_argument("--jaspar-dir", help="folder of per-motif .jaspar files (optional; copied)")
    j.add_argument("--sites-dir", help="folder of JASPAR per-motif <MA>.sites files (copied)")
    j.add_argument("--out", default="./Shared")

    u = sub.add_parser("unibind", help="build external-validation Shared/ from UniBind")
    u.add_argument("--tar", required=True)
    u.add_argument("--jaspar", required=True, help="an already-assembled Shared/ (for the TF->MA map + PWMs)")
    u.add_argument("--out", default="./Shared_unibind")
    u.add_argument("--kmin", type=int, default=7)
    u.add_argument("--kmax", type=int, default=14)
    u.add_argument("--all-versions", action="store_true")

    args = ap.parse_args()

    if args.cmd == "jaspar":
        os.makedirs(args.out, exist_ok=True)
        tdir = os.path.join(args.out, "transfac")
        if args.transfac_dir:
            print(f"[jaspar] copied {copy_dir(args.transfac_dir, tdir, ('*.transfac','*'))} transfac files")
        if args.transfac:
            print(f"[jaspar] split {split_combined_transfac(args.transfac, tdir)} motifs from combined TRANSFAC")
        if args.jaspar_dir:
            print(f"[jaspar] copied {copy_dir(args.jaspar_dir, os.path.join(args.out,'Jaspar_pwms'), ('*.jaspar',))} .jaspar files")
        if args.jaspar:
            print(f"[jaspar] split {split_combined_jaspar(args.jaspar, os.path.join(args.out,'Jaspar_pwms'))} .jaspar motifs")
        if args.sites_dir:
            print(f"[jaspar] copied {copy_dir(args.sites_dir, os.path.join(args.out,'sites'), ('*.sites',))} .sites files")
        if not os.path.isdir(tdir) or not glob.glob(os.path.join(tdir, '*')):
            ap.error("no transfac files assembled — pass --transfac-dir or --transfac")
        ann = load_annotations(args.out)
        miss = [m for m, a in ann.items() if not a["tf_family"] or not a["tf_class"]]
        print(f"[jaspar] {len(ann)} motifs annotated; {len(miss)} missing family/class.")
        print(f"[jaspar] Shared/ ready at {args.out}. Run: python -m motifgate --wdir {args.out} ...")

    elif args.cmd == "unibind":
        ann = load_annotations(args.jaspar)
        if not ann:
            ap.error(f"no transfac/ annotations under {args.jaspar}; run the 'jaspar' step first")
        os.makedirs(args.out, exist_ok=True)
        ex = os.path.join(args.out, "_unibind_extracted")
        os.makedirs(ex, exist_ok=True)
        print(f"[unibind] extracting {args.tar} ...")
        with tarfile.open(args.tar) as tf:
            tf.extractall(ex)
        flat = os.path.join(args.out, "_unibind_fasta"); os.makedirs(flat, exist_ok=True)
        for root, _, fnames in os.walk(ex):
            for fn in fnames:
                if fn.endswith((".fa", ".fasta", ".fas")):
                    os.replace(os.path.join(root, fn), os.path.join(flat, fn))
        for sub_ in ("transfac", "Jaspar_pwms"):
            srcd = os.path.join(args.jaspar, sub_)
            if os.path.isdir(srcd):
                shutil.copytree(srcd, os.path.join(args.out, sub_), dirs_exist_ok=True)
        unibind_to_sites(flat, args.out, ann, args.kmin, args.kmax, all_versions=args.all_versions)
        print(f"[unibind] external Shared/ ready at {args.out}. Run: python -m motifgate --wdir {args.out} ...")


if __name__ == "__main__":
    main()
