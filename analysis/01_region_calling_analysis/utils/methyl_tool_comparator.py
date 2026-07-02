import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import pybedtools
import pybedtools as pbt
import yaml
from plotly.subplots import make_subplots
from pybedtools import BedTool
from tqdm.auto import tqdm

# Historical import notes lived here during the benchmark phase.


from methylseg import SampleInfo
from methylseg import MethylDataPrep
from methylseg import MethylationStates
from methylseg import (
    MethylSegPathway as MethylSegToolPathway,
    MethylStateAssignmentMethod,
)

CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR
while not (PROJECT_ROOT / "repo_paths.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

REPO_HELPER_FILES = (
    PROJECT_ROOT
    / "analysis"
    / "01_region_calling_analysis"
    / "utils"
    / "helper_files"
)
FILES = REPO_HELPER_FILES if REPO_HELPER_FILES.exists() else CODE_DIR / "helper_files"
CANONICAL_CHROMS = frozenset(
    [f"chr{i}" for i in range(1, 23)]
    + ["chrX", "chrY"]
    + [str(i) for i in range(1, 23)]
    + ["X", "Y"]
)


@dataclass(frozen=True)
class SharedPrepArtifacts:
    wgbs_tsv: Path
    hm450k_bed: Path
    wgbs_450k_intersect_tsv: Path
    wgbs_beta: Path
    hm450k_beta: Path
    wgbs_meth_ref: Path
    hm450k_meth_ref: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.wgbs_tsv,
            self.hm450k_bed,
            self.wgbs_450k_intersect_tsv,
            self.wgbs_beta,
            self.hm450k_beta,
            self.wgbs_meth_ref,
            self.hm450k_meth_ref,
        )


class MemorySampler:
    def __init__(self, proc, interval=0.05):
        self.proc = proc
        self.interval = interval
        self.peak_rss = proc.memory_info().rss
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                rss = self.proc.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()


class RunStats:
    def __init__(
        self, tool_name: str, runtime_sec: float = None, mem_peak_mb: float = None
    ):
        self.tool_name = tool_name
        self.runtime_sec = runtime_sec
        self.mem_peak_mb = mem_peak_mb

        if self.runtime_sec is None or self.mem_peak_mb is None:
            self.cache_used = True
        else:
            self.cache_used = False

    def __str__(self):
        if self.cache_used:
            return f"Tool: {self.tool_name} | " f"Cache used "
        return (
            f"Tool: {self.tool_name} | "
            f"Runtime: {self.runtime_sec:.2f} sec | "
            f"Peak Memory: {self.mem_peak_mb:.1f} MB"
        )

    def __repr__(self):
        return self.__str__()

    def to_csv(self, out_dir: Path) -> Path:
        out_file = out_dir / f"{self.tool_name}_run_stats.csv"

        stats = pd.DataFrame(
            {
                "tool_name": [self.tool_name],
                "runtime_sec": [self.runtime_sec],
                "mem_peak_mb": [self.mem_peak_mb],
                "cache_used": [self.cache_used],
            }
        )

        stats.to_csv(out_file, index=False)
        return out_file


class MethylToolPathway:

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate: bool = False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        self.sample_id = sample_id
        self.meth_file = Path(meth_file)
        self.genome = genome
        self.out_dir = Path(out_dir)
        self.force_recreate = force_recreate
        self.print_logs = print_logs
        self.shared_prep = shared_prep
        self.run_stats: List[RunStats] = []

        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        self.log_file = self.out_dir / sample_id / "logs" / f"pipeline_{time_str}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        job_log_dir = self.log_file.parent / "job_logs"
        job_log_dir.mkdir(parents=True, exist_ok=True)
        # clear log file
        with open(self.log_file, "w") as log_fh:
            log_fh.write("")

    def _make_tmp_output_path(self, out_path: Path) -> Path:
        return out_path.parent / f".{out_path.name}.tmp.{os.getpid()}.{time.time_ns()}"

    def _cleanup_temp_path(self, path: Path) -> None:
        if path.is_symlink() or path.exists():
            path.unlink()

    def _atomic_write_path(
        self, out_path: Path, writer: Callable[[Path], None]
    ) -> Path:
        out_path = Path(out_path)
        if out_path.exists() and not self.force_recreate:
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._make_tmp_output_path(out_path)
        self._cleanup_temp_path(tmp_path)

        try:
            writer(tmp_path)
            if not tmp_path.exists() and not tmp_path.is_symlink():
                raise FileNotFoundError(
                    f"Temporary output was not created for {out_path}"
                )
            tmp_path.replace(out_path)
        finally:
            self._cleanup_temp_path(tmp_path)

        return out_path

    def _resolve_shared_prep(self) -> SharedPrepArtifacts | None:
        if self.shared_prep is None:
            return None

        prepare = getattr(self.shared_prep, "prepare", None)
        if callable(prepare):
            return prepare()

        return self.shared_prep

    def _materialize_shared_file(self, shared_path: Path, out_path: Path) -> Path:
        shared_path = Path(shared_path)
        if not shared_path.exists():
            raise FileNotFoundError(f"Missing shared prep file: {shared_path}")

        def link_or_copy(tmp_path: Path) -> None:
            try:
                os.symlink(shared_path.resolve(), tmp_path)
            except OSError:
                shutil.copy2(shared_path, tmp_path)

        return self._atomic_write_path(out_path, link_or_copy)

    def _sample_dir(self) -> Path:
        return Path(self.out_dir) / self.sample_id

    def _prep_dir(self) -> Path:
        prep_dir = self._sample_dir() / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)
        return prep_dir

    def _prep_config_path(self) -> Path:
        return self._prep_dir() / "config.yaml"

    @staticmethod
    def _cpg_coord_cols() -> list[str]:
        return ["CpG_chrm", "CpG_beg", "CpG_end"]

    @staticmethod
    def _sanitize_yaml_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "value") and hasattr(value, "name"):
            return MethylToolPathway._sanitize_yaml_value(value.value)
        if isinstance(value, dict):
            return {
                str(key): MethylToolPathway._sanitize_yaml_value(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [MethylToolPathway._sanitize_yaml_value(val) for val in value]
        return value

    def _write_prep_config(self, payload: dict[str, Any]) -> Path:
        sanitized_payload = self._sanitize_yaml_value(payload)

        def write_yaml(tmp_out: Path) -> None:
            with open(tmp_out, "w") as out_fh:
                yaml.safe_dump(sanitized_payload, out_fh, sort_keys=False)

        return self._atomic_write_path(self._prep_config_path(), write_yaml)

    @classmethod
    def _finalize_cpg_tracking_df(
        cls,
        df: pd.DataFrame,
        *,
        meth_col: str | None = None,
        coverage_col: str | None = None,
        beta_col: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df["CpG_chrm"] = df["CpG_chrm"].astype(str)
        df["CpG_beg"] = pd.to_numeric(df["CpG_beg"], errors="coerce").astype("Int64")
        df["CpG_end"] = pd.to_numeric(df["CpG_end"], errors="coerce").astype("Int64")

        if meth_col is not None and meth_col in df.columns:
            df[meth_col] = pd.to_numeric(df[meth_col], errors="coerce")
        if coverage_col is not None and coverage_col in df.columns:
            df[coverage_col] = pd.to_numeric(df[coverage_col], errors="coerce")

        if beta_col is not None and beta_col in df.columns:
            df["beta"] = pd.to_numeric(df[beta_col], errors="coerce")
        elif meth_col is not None and coverage_col is not None:
            coverage = pd.to_numeric(df[coverage_col], errors="coerce")
            meth = pd.to_numeric(df[meth_col], errors="coerce")
            df["beta"] = np.where(coverage > 0, meth / coverage, np.nan)
        else:
            df["beta"] = np.nan

        df = df.dropna(subset=["CpG_beg", "CpG_end"]).reset_index(drop=True)
        df["CpG_beg"] = df["CpG_beg"].astype(np.int64)
        df["CpG_end"] = df["CpG_end"].astype(np.int64)
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
        return df

    @classmethod
    def _build_plot_universe_df(cls, tracking_df: pd.DataFrame) -> pd.DataFrame:
        if tracking_df is None or tracking_df.empty:
            return pd.DataFrame(columns=cls._cpg_coord_cols() + ["beta"])

        beta = pd.to_numeric(tracking_df["beta"], errors="coerce")
        plot_mask = beta.notna() & np.isfinite(beta) & beta.between(0.0, 1.0)
        plot_df = tracking_df.loc[plot_mask, cls._cpg_coord_cols() + ["beta"]].copy()
        return plot_df.sort_values(cls._cpg_coord_cols()).reset_index(drop=True)

    @classmethod
    def _load_wgbs_tracking_df(cls, wgbs_tsv: Path) -> pd.DataFrame:
        wgbs_df = pd.read_csv(
            wgbs_tsv,
            sep="\t",
            header=None,
            names=["CpG_chrm", "CpG_beg", "CpG_end", "meth", "coverage"],
            low_memory=False,
        )
        return cls._finalize_cpg_tracking_df(
            wgbs_df,
            meth_col="meth",
            coverage_col="coverage",
        )

    @classmethod
    def _load_intersect_tracking_df(cls, intersect_tsv: Path) -> pd.DataFrame:
        intersect_df = pd.read_csv(
            intersect_tsv,
            sep="\t",
            header=None,
            names=[
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "meth",
                "coverage",
                "chr2",
                "start2",
                "end2",
                "probe",
            ],
            low_memory=False,
        )
        return cls._finalize_cpg_tracking_df(
            intersect_df,
            meth_col="meth",
            coverage_col="coverage",
        )

    def _filter_to_canonical_chromosomes(self, tsv_path: Path) -> None:
        filtered_path = tsv_path.with_suffix(tsv_path.suffix + ".canonical_tmp")
        with open(tsv_path) as in_fh, open(filtered_path, "w") as out_fh:
            for line in in_fh:
                if line.partition("	")[0] in CANONICAL_CHROMS:
                    out_fh.write(line)
        filtered_path.replace(tsv_path)

    def _create_readable_meth_file(
        self, out_tsv: Path, canonical_chromosomes_only: bool = True
    ) -> Path:
        """
        If input is .beta: run wgbstools view --genome <genome> <beta> -o <out_tsv>
        Else: copy to out_tsv
        """
        out_tsv = Path(out_tsv)

        def write_readable_file(tmp_out: Path) -> None:
            if self.meth_file.suffix == ".beta":
                cmd = [
                    "wgbstools",
                    "view",
                    "--genome",
                    self.genome,
                    str(self.meth_file),
                    "-o",
                    str(tmp_out),
                ]
                self._run(cmd)
            elif ".bed.gz" in "".join(self.meth_file.suffixes):
                cmd = [
                    "gunzip",
                    "-c",
                    str(self.meth_file),
                ]
                raw_tmp = self._make_tmp_output_path(
                    tmp_out.with_suffix(tmp_out.suffix + ".unsorted")
                )
                try:
                    self._run(cmd, stdout_path=raw_tmp.resolve())

                    with open(raw_tmp) as fh:
                        first_line = fh.readline().rstrip("\n")
                    n_cols = len(first_line.split("	"))

                    if n_cols == 5:
                        unordered_df = pd.read_csv(
                            raw_tmp,
                            sep="	",
                            index_col=None,
                            header=None,
                            names=["chrom", "start", "end", "cov", "meth_percent"],
                            usecols=[0, 1, 2, 3, 4],
                        )
                        unordered_df["meth"] = (
                            unordered_df["meth_percent"] / 100.0 * unordered_df["cov"]
                        )
                    elif n_cols == 6:
                        unordered_df = pd.read_csv(
                            raw_tmp,
                            sep="	",
                            index_col=None,
                            header=None,
                            names=["chrom", "start", "end", "beta", "cov"],
                            usecols=[0, 1, 2, 3, 4],
                        )
                        unordered_df["meth"] = (
                            unordered_df["beta"] * unordered_df["cov"]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported .bed.gz format with {n_cols} columns in {self.meth_file}"
                        )

                    unordered_df["meth"] = unordered_df["meth"].round().astype(int)
                    unordered_df = unordered_df[
                        ["chrom", "start", "end", "meth", "cov"]
                    ]
                    unordered_df.to_csv(tmp_out, header=None, sep="	", index=False)
                finally:
                    self._cleanup_temp_path(raw_tmp)
            else:
                cmd = ["cp", str(self.meth_file), str(tmp_out)]
                self._run(cmd)

            if canonical_chromosomes_only:
                self._filter_to_canonical_chromosomes(tmp_out)

        return self._atomic_write_path(out_tsv, write_readable_file)

    def _wgbs_cov_to_beta_file(self, wgbs_tsv: Path, out_beta: Path) -> Path:
        def write_beta_file(tmp_out: Path) -> None:
            wgbs_data_df = pd.read_csv(
                wgbs_tsv,
                sep="	",
                header=None,
                names=["chrom", "start", "end", "meth", "coverage"],
            )
            wgbs_data_df["beta"] = wgbs_data_df["meth"] / wgbs_data_df["coverage"]
            wgbs_data_df[["chrom", "start", "end", "beta"]].to_csv(
                tmp_out,
                sep="	",
                index=False,
            )

        return self._atomic_write_path(out_beta, write_beta_file)

    def _make_meth_ref(
        self,
        meth_file: Path,
        meth_ref: Path,
        resolution: str,
    ) -> Path:
        def write_meth_ref(tmp_out: Path) -> None:
            MethylDataPrep(
                meth_file=meth_file,
                sample_id=self.sample_id,
                resolution=resolution,
            ).write_prepared_tsv(tmp_out)

        return self._atomic_write_path(meth_ref, write_meth_ref)

    def _write_log(self, message: str) -> None:
        """Append a message to the log file with a timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_file, "a") as log_fh:
            log_message = f"[{timestamp}] {message}\n"
            if self.print_logs:
                print(log_message.strip())
            log_fh.write(log_message)

    def _run(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> dict:
        """
        Run a command, tracking wall-clock runtime and *peak* RSS memory
        of the child process (not the parent Python process).
        """
        start = time.perf_counter()

        stdout_handle = None
        stderr_handle = None

        cmd_name = Path(cmd[0]).name
        log_base = (
            Path(self.out_dir)
            / self.sample_id
            / "logs"
            / "job_logs"
            / f"{cmd_name}_{start:.0f}.log"
        )
        log_base.parent.mkdir(parents=True, exist_ok=True)

        # stdout
        if stdout_path is None:
            stdout_path = log_base.with_suffix(".stdout")
        elif not isinstance(stdout_path, Path):
            raise TypeError("stdout_path must be a Path or None")

        stdout_handle = open(stdout_path, "wb")

        # stderr
        if stderr_path is None:
            stderr_path = log_base.with_suffix(".stderr")
        elif not isinstance(stderr_path, Path):
            raise TypeError("stderr_path must be a Path or None")

        stderr_handle = open(stderr_path, "wb")

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

        peak_rss = 0  # bytes

        try:
            child = psutil.Process(p.pid)

            # poll until process finishes
            while p.poll() is None:
                try:
                    rss = child.memory_info().rss
                    for c in child.children(recursive=True):
                        rss += c.memory_info().rss
                    peak_rss = max(peak_rss, rss)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.1)  # polling interval (100 ms)
        finally:
            returncode = p.wait()
            if stdout_handle:
                stdout_handle.close()
            if stderr_handle:
                stderr_handle.close()

        end = time.perf_counter()

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        runtime = end - start
        peak_mem_mb = peak_rss / 1e6

        self._write_log(
            f"Command {' '.join(cmd)} | "
            f"Output: {str(stdout_path) if stdout_path else 'stdout'} | "
            f"Error: {str(stderr_path) if stderr_path else 'stderr'} | "
            f"time={runtime:.2f}s | "
            f"peak_mem={peak_mem_mb:.1f} MB"
        )

        return {
            "runtime_sec": runtime,
            "mem_peak_mb": peak_mem_mb,
        }

    def run_data_prep(
        self,
    ) -> dict:
        raise NotImplementedError

    def run_tool(
        self,
    ) -> dict:
        raise NotImplementedError

    def run(self):
        prep_outputs = self.run_data_prep()
        tool_outputs = self.run_tool()
        file_path = Path(self.out_dir) / self.sample_id / "run_stats"
        file_path.mkdir(parents=True, exist_ok=True)

        for run_stat in self.run_stats:
            out_file = file_path / f"{run_stat.tool_name}_run_stats.csv"
            if not out_file.exists() or self.force_recreate:
                run_stat.to_csv(file_path)
        return prep_outputs, tool_outputs


class MicroArrayPathway(MethylToolPathway):

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )

    def _create_hm450k_bed(
        self,
        hm450k_bed: Path,
    ) -> Path:
        """
        Run {tool_name}_data_prep.r to generate an HM450K BED
        """
        hm450k_bed = Path(hm450k_bed)

        r_script = FILES / Path("microarray_data_prep.r")
        if not r_script.exists():
            raise FileNotFoundError(f"Missing R script: {r_script}")

        chain_file = FILES / Path(f"hg19ToHg38.over.chain")
        if not chain_file.exists():
            raise FileNotFoundError(f"Missing chain file: {chain_file}")

        def write_hm450k_bed(tmp_out: Path) -> None:
            cmd = [
                "Rscript",
                str(r_script),
                "--genome",
                self.genome,
                "--chain-file",
                str(chain_file),
                "--out-bed",
                str(tmp_out),
            ]
            self._run(cmd)

        return self._atomic_write_path(hm450k_bed, write_hm450k_bed)

    def _bedtools_intersect_450k(
        self, wgbs_tsv: Path, hm450k_bed: Path, out_tsv: Path
    ) -> Path:
        def write_intersect(tmp_out: Path) -> None:
            a = pybedtools.BedTool(str(wgbs_tsv))
            b = pybedtools.BedTool(str(hm450k_bed))
            a.intersect(b, wa=True, wb=True).saveas(str(tmp_out))

        return self._atomic_write_path(out_tsv, write_intersect)

    def _format_intersect_as_450k_beta(
        self, intersect_file: Path, out_tsv: Path
    ) -> Path:
        def write_hm450k_beta(tmp_out: Path) -> None:
            df = pd.read_csv(
                intersect_file,
                sep="	",
                header=None,
                names=[
                    "chr",
                    "start",
                    "end",
                    "meth",
                    "cov",
                    "chr2",
                    "start2",
                    "end2",
                    "probe",
                ],
            )
            df = df[df["cov"] > 0].copy()
            df["beta"] = df["meth"] / df["cov"]
            beta_df = df[["chr", "start", "end", "beta", "probe"]].dropna(
                subset=["beta"],
                how="any",
            )
            beta_df.to_csv(tmp_out, sep="	", index=False, header=None)

        return self._atomic_write_path(out_tsv, write_hm450k_beta)


class SharedPrepManager(MicroArrayPathway):
    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate: bool = False,
        print_logs: bool = True,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep=None,
        )
        self.shared_prep_dir = Path(self.out_dir) / self.sample_id / "shared_prep"
        self._outputs: SharedPrepArtifacts | None = None

    def prepare(self) -> SharedPrepArtifacts:
        if self._outputs is not None and all(
            path.exists() for path in self._outputs.all_paths()
        ):
            self._write_log(
                f"Reusing shared prep artifacts from {self.shared_prep_dir}"
            )
            return self._outputs

        shared_dir = self.shared_prep_dir
        shared_dir.mkdir(parents=True, exist_ok=True)
        self._write_log(f"Building shared prep artifacts in {shared_dir}")

        wgbs_tsv = self._create_readable_meth_file(shared_dir / "wgbs.tsv")
        hm450k_bed = self._create_hm450k_bed(
            shared_dir / f"HM450K_{self.genome}_locations.bed"
        )
        wgbs_450k_intersect_tsv = self._bedtools_intersect_450k(
            wgbs_tsv,
            hm450k_bed,
            shared_dir / "wgbs_450k_intersect.tsv",
        )
        wgbs_beta = self._wgbs_cov_to_beta_file(wgbs_tsv, shared_dir / "wgbs.beta")
        hm450k_beta = self._format_intersect_as_450k_beta(
            wgbs_450k_intersect_tsv,
            shared_dir / "450k.beta",
        )
        wgbs_meth_ref = self._make_meth_ref(
            wgbs_tsv,
            shared_dir / "wgbs_meth_ref.tsv",
            resolution="wgbs",
        )
        hm450k_meth_ref = self._make_meth_ref(
            hm450k_beta,
            shared_dir / "450k_meth_ref.tsv",
            resolution="450k",
        )

        self._outputs = SharedPrepArtifacts(
            wgbs_tsv=wgbs_tsv,
            hm450k_bed=hm450k_bed,
            wgbs_450k_intersect_tsv=wgbs_450k_intersect_tsv,
            wgbs_beta=wgbs_beta,
            hm450k_beta=hm450k_beta,
            wgbs_meth_ref=wgbs_meth_ref,
            hm450k_meth_ref=hm450k_meth_ref,
        )
        return self._outputs


class DNMToolsPathway(MicroArrayPathway):

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )

    def _wgbs_to_dnmtools_format(self, wgbs_tsv: Path, out_meth: Path) -> Path:
        """
        Convert a WGBS table (chrom, start, end, meth, coverage) to DNMTools format:
        chr  pos  strand  context  meth_level  coverage
        with meth_level = meth / coverage
        """
        if out_meth.exists() and not self.force_recreate:
            return out_meth
        out_meth.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            wgbs_tsv,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "meth", "coverage"],
        )

        df = df[df["coverage"] > 0].copy()
        df["meth_level"] = df["meth"] / df["coverage"]

        dnm_df = pd.DataFrame(
            {
                "chr": df["chrom"],
                "pos": df["start"],  # CpG position
                "strand": "+",
                "context": "CpG",
                "meth_level": df["meth_level"],
                "coverage": df["coverage"],
            }
        ).dropna(subset=["meth_level"])

        dnm_df.to_csv(out_meth, sep="\t", index=False, header=None)
        return out_meth

    def _format_450k_like_dnmtools(self, wgbs_450k_tsv: Path, out_tsv: Path) -> Path:
        """
        Your notebook reads bedtools output columns:
        chr start end meth cov chr2 start2 end2 probe
        and writes DNMTools-like columns (no header).
        """
        if out_tsv.exists() and not self.force_recreate:
            return out_tsv
        out_tsv.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            wgbs_450k_tsv,
            sep="\t",
            header=None,
            names=[
                "chr",
                "start",
                "end",
                "meth",
                "cov",
                "chr2",
                "start2",
                "end2",
                "probe",
            ],
        )

        # df.loc[df["cov"] == 0, "cov"] = pd.NA
        df = df[df["cov"] > 0].copy()
        df["meth_level"] = df["meth"] / df["cov"]

        dnm_df = pd.DataFrame(
            {
                "chr": df["chr"],
                "pos": df["start"],
                "strand": "+",
                "context": "CpG",
                "meth_level": df["meth_level"],
                "coverage": df["cov"],
            }
        ).dropna(subset=["meth_level", "coverage"], how="any")

        dnm_df.to_csv(out_tsv, sep="\t", index=False, header=None)
        return out_tsv

    def run_data_prep(
        self,
    ) -> dict:
        """
        End-to-end replacement for your notebook:
        - wgbstools view (if .beta) or copy
        - convert to DNMTools .meth
        - optionally create 450k subset using R+bedtools and format it
        """
        prep_dir = self._prep_dir()

        shared_prep = self._resolve_shared_prep()
        if shared_prep is not None:
            wgbs_tsv = self._materialize_shared_file(
                shared_prep.wgbs_tsv,
                prep_dir / "wgbs.tsv",
            )
            self._materialize_shared_file(
                shared_prep.hm450k_bed,
                prep_dir / f"HM450K_{self.genome}_locations.bed",
            )
            wgbs_450k_tsv = self._materialize_shared_file(
                shared_prep.wgbs_450k_intersect_tsv,
                prep_dir / "wgbs_450k.tsv",
            )
        else:
            wgbs_tsv = self._create_readable_meth_file(prep_dir / "wgbs.tsv")
            hm450k_bed_path = self._create_hm450k_bed(
                hm450k_bed=prep_dir / f"HM450K_{self.genome}_locations.bed",
            )
            wgbs_450k_tsv = self._bedtools_intersect_450k(
                wgbs_tsv,
                hm450k_bed_path,
                prep_dir / "wgbs_450k.tsv",
            )

        dnm_meth = self._wgbs_to_dnmtools_format(wgbs_tsv, prep_dir / "dnmtools.meth")
        dnm_450k_tsv = self._format_450k_like_dnmtools(
            wgbs_450k_tsv,
            prep_dir / "450k.tsv",
        )

        return {
            "wgbs_tsv": wgbs_tsv,
            "dnmtools_meth": dnm_meth,
            "wgbs_450k_tsv": wgbs_450k_tsv,
            "dnmtools_450k_tsv": dnm_450k_tsv,
        }

    def run_tool(
        self,
    ):
        input_file = Path(self.out_dir) / self.sample_id / "prep" / f"dnmtools.meth"
        output_dir = Path(self.out_dir) / self.sample_id / "out"
        hm450k_file = Path(self.out_dir) / self.sample_id / "prep" / f"450k.tsv"

        output_dir.mkdir(parents=True, exist_ok=True)
        pmd_output_file = output_dir / f"dnmtools_PMDs.bed"

        pmd_cmd = [
            "dnmtools",
            "pmd",
            "-i",
            "1000",
            "-o",
            str(pmd_output_file),
            str(input_file),
        ]
        pmd_run_stats = {}
        if not pmd_output_file.exists() or self.force_recreate:
            pmd_run_stats = self._run(pmd_cmd)

        array_output_file = output_dir / f"arraymode.dnmtools_PMDs.bed"
        pmd_array_cmd = [
            "dnmtools",
            "pmd",
            "-i",
            "1000",
            "-a",
            "-d",
            "100000",
            "-b",
            "20000",
            "-o",
            str(array_output_file),
            str(hm450k_file),
        ]
        array_run_stats = {}
        if not array_output_file.exists() or self.force_recreate:
            array_run_stats = self._run(pmd_array_cmd)

        pmr_output_file = output_dir / f"pmr.dnmtools_PMDs.bed"

        pmr_cmd = [
            "dnmtools",
            "hmr",
            "-partial",
            "-o",
            str(pmr_output_file),
            str(input_file),
        ]
        pmr_run_stats = {}
        if not pmr_output_file.exists() or self.force_recreate:
            pmr_run_stats = self._run(pmr_cmd)

        self.run_stats.append(RunStats(tool_name="DNMTools_PMD", **pmd_run_stats))
        self.run_stats.append(
            RunStats(tool_name="DNMTools_Array_PMD", **array_run_stats)
        )
        self.run_stats.append(RunStats(tool_name="DNMTools_PMR", **pmr_run_stats))

        prep_dir = self._prep_dir()
        self._write_prep_config(
            {
                "tool_name": "DNMTools",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": prep_dir / "wgbs.tsv",
                    "wgbs_450k_tsv": prep_dir / "wgbs_450k.tsv",
                    "dnmtools_meth": prep_dir / "dnmtools.meth",
                    "dnmtools_450k_tsv": prep_dir / "450k.tsv",
                },
                "prep_filters": {
                    "wgbs": [
                        "coverage > 0",
                        "drop rows with non-finite meth_level before writing dnmtools.meth",
                    ],
                    "hm450k": [
                        "coverage > 0",
                        "drop rows with non-finite meth_level before writing 450k.tsv",
                    ],
                },
                "tool_commands": {
                    "pmd": pmd_cmd,
                    "array_pmd": pmd_array_cmd,
                    "pmr": pmr_cmd,
                },
            }
        )

        return {
            "pmd_bed": str(pmd_output_file),
            "pmd_array_bed": str(array_output_file),
            "pmr_bed": str(pmr_output_file),
        }


class MethylseekRPathway(MethylToolPathway):

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )

    def run_data_prep(
        self,
    ) -> dict:
        prep_dir = self._prep_dir()

        shared_prep = self._resolve_shared_prep()
        if shared_prep is not None:
            wgbs_tsv = self._materialize_shared_file(
                shared_prep.wgbs_tsv,
                prep_dir / "wgbs.tsv",
            )
        else:
            wgbs_tsv = self._create_readable_meth_file(prep_dir / "wgbs.tsv")

        return {
            "wgbs_tsv": wgbs_tsv,
        }

    def run_tool(
        self,
    ) -> dict:
        r_script = FILES / Path("run_methylseekr.r")
        if not r_script.exists():
            raise FileNotFoundError(f"Missing R script: {r_script}")

        cmd = [
            "Rscript",
            str(r_script),
            "--genome",
            self.genome,
            "--sample",
            str(self.sample_id),
            "--out_dir",
            str(self.out_dir),
        ]
        out_bed = (
            Path(self.out_dir)
            / Path(self.sample_id)
            / Path("out")
            / Path(f"methylseekr_PMDs.bed")
        )
        out_bed.parent.mkdir(parents=True, exist_ok=True)
        log_file = (
            Path(self.out_dir)
            / Path(self.sample_id)
            / Path("logs")
            / Path("job_logs")
            / Path("methylseekr.log")
        )
        methylseekr_run_stats = {}
        if not out_bed.exists() or self.force_recreate:
            methylseekr_run_stats = self._run(cmd)

            if not log_file.exists():
                raise FileNotFoundError(f"Missing MethylSeekR log file: {log_file}")
            if self.print_logs:
                for line in open(log_file):
                    print(line.strip())
        self.run_stats.append(
            RunStats(tool_name="MethylSeekR", **methylseekr_run_stats)
        )
        self._write_prep_config(
            {
                "tool_name": "MethylSeekR",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": self._prep_dir() / "wgbs.tsv",
                },
                "prep_filters": {
                    "wgbs": ["coverage > 0", "meth <= coverage"],
                },
                "tool_runtime_settings": {
                    "script_path": r_script,
                },
                "tool_commands": {
                    "run_methylseekr": cmd,
                },
            }
        )
        return {
            "pmd_bed": str(out_bed),
            "log_file": str(log_file),
        }


class MethylSegPathway(MicroArrayPathway):
    CLEAN_MIN_CPGS = 6
    CLEAN_MIN_REGION_LENGTH = 5000
    CLEAN_MERGE_GAP_BP = 100_000
    EXPANSION_MAX_GAP_BP = 100_000

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
        methyl_seg_config=FILES / Path("methyl_seg_config.yaml"),
        wgbs_window_specs=[
            (500, "500bp"),
            # (10_000, "10kb"),
            (40_000, "40kb"),
            (450_000, "450kb"),
            # (500_000, "500kb"),
            # (1_000_000, "1Mb"),
        ],
        hm450_window_specs=[
            # (5000, "5kb"),
            # (10_000, "10kb"),
            (40_000, "40kb"),
            (450_000, "450kb"),
            # (500_000, "500kb"),
            # (1_000_000, "1Mb"),
        ],
        min_coverage=10,
        n_states=4,
        int_low_cutoff=0.2,
        int_high_cutoff=0.7,
        high_cutoff=0.7,
        train_sample_info: SampleInfo | None = None,
        train_sample=None,
        train_sample_file=None,
        train_chrom="chr1",
        random_state=42,
        hm450_hmm_type="ct",
        hm450_hmm_params={
            "n_emissions": 4,
            "holding_time_guess": 1_500_000,
            "algorithm": "forward-backward",
            "max_iter": 25,
            "tol": 1e-2,
        },
        # hm450_hmm_type="sticky",
        # hm450_hmm_params={},
        wgbs_hmm_type="sticky",
        wgbs_hmm_params={
            "stay_prob": 0.99995,
            "emission_mismatch_prob": 0.45,
            "fit_transitions": False,
        },
        fit_methyl_seg=True,
        state_assignment_method=MethylStateAssignmentMethod.KMEANS,
        merge_with_intermediate=True,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )
        if methyl_seg_config is None and not fit_methyl_seg:
            raise ValueError("Must provide methyl_seg_config if not fitting MethylSeg")
        self.methyl_seg_config = methyl_seg_config
        self.wgbs_window_specs = wgbs_window_specs
        self.hm450_window_specs = hm450_window_specs
        self.min_coverage = min_coverage
        self.n_states = n_states
        self.int_low_cutoff = int_low_cutoff
        self.int_high_cutoff = int_high_cutoff
        self.high_cutoff = high_cutoff
        if (
            train_sample_file is None
            and train_sample is not None
            or train_sample_file is not None
            and train_sample is None
        ):
            raise ValueError(
                "train_sample and train_sample_file must both be provided or both be None"
            )
        if train_sample_info is None and train_sample_file is not None:
            self.train_sample_info, _ = MethylSegToolPathway.prepare_sample_info(
                sample_name=train_sample,
                sample_file=Path(train_sample_file),
                resolution="auto",
            )
            self.train_sample = train_sample
        elif train_sample_info is not None:
            self.train_sample_info = train_sample_info
            self.train_sample = train_sample_info.sample_id
        else:
            self.train_sample_info = None
            self.train_sample = None
        self.train_chrom = train_chrom
        self.random_state = random_state
        self.hm450_hmm_type = hm450_hmm_type
        self.hm450_hmm_params = hm450_hmm_params
        self.wgbs_hmm_type = wgbs_hmm_type
        self.wgbs_hmm_params = wgbs_hmm_params
        self.fit_methyl_seg = fit_methyl_seg
        self.hm450_removed_cpgs = pd.DataFrame()
        self.wgbs_removed_cpgs = pd.DataFrame()
        self.methylseg_clean_min_cpgs = self.CLEAN_MIN_CPGS
        self.methylseg_clean_min_region_length = self.CLEAN_MIN_REGION_LENGTH
        self.methylseg_clean_merge_gap_bp = self.CLEAN_MERGE_GAP_BP
        self.merge_with_intermediate = merge_with_intermediate
        self.methylseg_expansion_merge_gap_bp = self.EXPANSION_MAX_GAP_BP
        if (
            not self.fit_methyl_seg
            and state_assignment_method != MethylStateAssignmentMethod.DEFINITION
        ):
            raise ValueError(
                "If not fitting MethylSeg, state_assignment_method must be 'definition'"
            )
        self.state_assignment_method = state_assignment_method

    def _wgbs_cov_to_beta_file(self, wgbs_tsv: Path, out_beta: Path):
        if out_beta.exists() and not self.force_recreate:
            return out_beta
        out_beta.parent.mkdir(parents=True, exist_ok=True)
        wgbs_data_df = pd.read_csv(
            wgbs_tsv,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "meth", "coverage"],
        )
        wgbs_data_df["beta"] = wgbs_data_df["meth"] / wgbs_data_df["coverage"]
        wgbs_data_df[
            [
                "chrom",
                "start",
                "end",
                "beta",
            ]
        ].to_csv(out_beta, sep="\t", index=False)
        return out_beta

    def _make_meth_ref(
        self,
        meth_file: Path,
        meth_ref: Path,
        resolution: str,
    ) -> pd.DataFrame:
        if meth_ref.exists() and not self.force_recreate:
            return meth_ref
        return MethylDataPrep(
            meth_file=meth_file,
            sample_id=self.sample_id,
            resolution=resolution,
        ).write_prepared_tsv(meth_ref)

    @staticmethod
    def _annotate_removed_cpg_reason(
        removed_cpgs: pd.DataFrame, reason: str
    ) -> pd.DataFrame:
        annotated = removed_cpgs.copy()
        annotated["removed_reason"] = reason
        return annotated

    def _annotate_wgbs_removed_cpg_reasons(
        self, removed_cpgs: pd.DataFrame
    ) -> pd.DataFrame:
        annotated = removed_cpgs.copy()
        if annotated.empty:
            annotated["removed_reason"] = pd.Series(dtype="object")
            return annotated

        coverage = pd.to_numeric(annotated.get("coverage"), errors="coerce")
        annotated["removed_reason"] = np.where(
            coverage.lt(self.min_coverage),
            f"cov less than {self.min_coverage}",
            "low coverage like",
        )
        return annotated

    def _removed_cpgs_path(self, platform: str) -> Path:
        if platform not in {"wgbs", "hm450k"}:
            raise ValueError(f"Unsupported removed CpG platform: {platform}")
        return self._prep_dir() / f"{platform}_removed_cpgs.tsv"

    def _write_removed_cpgs(self, platform: str, removed_cpgs: pd.DataFrame) -> Path:
        out_path = self._removed_cpgs_path(platform)

        def write_removed_cpgs(tmp_out: Path) -> None:
            removed_cpgs.to_csv(tmp_out, sep="\t", index=False)

        return self._atomic_write_path(out_path, write_removed_cpgs)

    def load_removed_cpgs(self, platform: str) -> pd.DataFrame:
        removed_cpgs_path = self._removed_cpgs_path(platform)
        if not removed_cpgs_path.exists():
            return pd.DataFrame()
        return pd.read_csv(removed_cpgs_path, sep="\t")

    def _format_bedtools(self, intersect_file: Path, out_tsv: Path) -> Path:
        """
        Your notebook reads bedtools output columns:
        chr start end meth cov chr2 start2 end2 probe
        and writes DNMTools-like columns (no header).
        """
        if out_tsv.exists() and not self.force_recreate:
            return out_tsv
        out_tsv.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            intersect_file,
            sep="\t",
            header=None,
            names=[
                "chr",
                "start",
                "end",
                "meth",
                "cov",
                "chr2",
                "start2",
                "end2",
                "probe",
            ],
        )

        # df.loc[df["cov"] == 0, "cov"] = pd.NA
        df = df[df["cov"] > 0].copy()
        df["beta"] = df["meth"] / df["cov"]

        dnm_df = df[
            [
                "chr",
                "start",
                "end",
                "beta",
                "probe",
            ]
        ].dropna(subset=["beta"], how="any")

        dnm_df.to_csv(out_tsv, sep="\t", index=False, header=None)
        return out_tsv

    def run_data_prep(
        self,
    ) -> dict:
        prep_dir = self._prep_dir()

        shared_prep = self._resolve_shared_prep()
        if shared_prep is not None:
            wgbs_tsv = self._materialize_shared_file(
                shared_prep.wgbs_tsv,
                prep_dir / "wgbs.tsv",
            )
            self._materialize_shared_file(
                shared_prep.wgbs_beta,
                prep_dir / "wgbs.beta",
            )
            wgbs_meth_ref = self._materialize_shared_file(
                shared_prep.wgbs_meth_ref,
                prep_dir / "wgbs_meth_ref.tsv",
            )
            self._materialize_shared_file(
                shared_prep.hm450k_bed,
                prep_dir / f"HM450K_{self.genome}_locations.bed",
            )
            self._materialize_shared_file(
                shared_prep.wgbs_450k_intersect_tsv,
                prep_dir / "wgbs_450k_intersect.tsv",
            )
            self._materialize_shared_file(
                shared_prep.hm450k_beta,
                prep_dir / "450k.beta",
            )
            hm450k_ref_file = self._materialize_shared_file(
                shared_prep.hm450k_meth_ref,
                prep_dir / "450k_meth_ref.tsv",
            )
        else:
            wgbs_tsv = self._create_readable_meth_file(prep_dir / "wgbs.tsv")
            MethylToolPathway._wgbs_cov_to_beta_file(
                self, wgbs_tsv, prep_dir / "wgbs.beta"
            )
            wgbs_meth_ref = MethylToolPathway._make_meth_ref(
                self,
                wgbs_tsv,
                prep_dir / "wgbs_meth_ref.tsv",
                resolution="wgbs",
            )
            hm450k_bed_path = self._create_hm450k_bed(
                hm450k_bed=prep_dir / f"HM450K_{self.genome}_locations.bed",
            )
            intersect_file = self._bedtools_intersect_450k(
                wgbs_tsv,
                hm450k_bed_path,
                prep_dir / "wgbs_450k_intersect.tsv",
            )
            hm450k_beta_file = self._format_intersect_as_450k_beta(
                intersect_file,
                prep_dir / "450k.beta",
            )
            hm450k_ref_file = MethylToolPathway._make_meth_ref(
                self,
                hm450k_beta_file,
                prep_dir / "450k_meth_ref.tsv",
                resolution="450k",
            )

        return {
            "hm450k_meth_ref": str(hm450k_ref_file),
            "wgbs_meth_ref": str(wgbs_meth_ref),
        }

    def _run_methylseg_on_each_chromosome(
        self,
        methylseg: MethylSegToolPathway,
        sample_info: SampleInfo,
        tool_suffix: str = "",
    ) -> tuple[SampleInfo, list[str]]:
        all_chroms = sorted(set(sample_info.meth_data["CpG_chrm"].dropna()) - {"chrM"})
        if all_chroms:
            segmented_sample_info = MethylSegToolPathway.subset_sample_info_by_chroms(
                sample_info=sample_info,
                chroms=all_chroms,
            )
        else:
            segmented_sample_info = SampleInfo(
                sample_id=sample_info.sample_id,
                meth_data=sample_info.meth_data.iloc[0:0].copy(),
            )

        proc = psutil.Process(os.getpid())

        start = time.perf_counter()
        peak_rss = proc.memory_info().rss

        sampler = MemorySampler(proc, interval=0.05)
        sampler.start()

        if self.fit_methyl_seg:
            methylseg.fit_pathway()
            self._write_log("Fitting done, starting segmentation...")
            if self.state_assignment_method == MethylStateAssignmentMethod.DEFINITION:
                methylseg.analyzer.pretty_print_rules()
        else:
            methylseg.analyzer.set_state_cutoffs_from_yaml(self.methyl_seg_config)

        states = MethylationStates._member_names_
        cache_used = False
        summary_paths: list[str] = []

        try:
            if all_chroms:
                bed_paths = [
                    Path(methylseg.out_dir)
                    / f"segments_{chrom}_{segmented_sample_info.sample_id}_{state}.bed"
                    for chrom in all_chroms
                    for state in states
                ]
                summary_file_dir = Path(methylseg.out_dir) / "summary_files"
                summary_paths = self._collect_methylseg_output_paths(
                    output_dir=Path(methylseg.out_dir),
                    sample_info=segmented_sample_info,
                )
                required_summary_paths = [
                    summary_file_dir / f"segments_raw_{state}.bed" for state in states
                ] + [
                    summary_file_dir / f"segments_cleaned_{state}.bed"
                    for state in states
                ]

                if all(p.exists() for p in bed_paths) and not self.force_recreate:
                    if all(path.exists() for path in required_summary_paths):
                        cache_used = True
                    else:
                        methylseg._write_summary_files(
                            raw_regions_df=None,
                            sample_id=segmented_sample_info.sample_id,
                            clean_regions=True,
                            chroms=all_chroms,
                            write_raw_summaries=False,
                        )
                        if all(path.exists() for path in required_summary_paths):
                            cache_used = True
                        else:
                            summary_paths = methylseg.run_on_all_chroms(
                                sample_info=sample_info,
                                chroms=all_chroms,
                                min_probes=5,
                            )
                else:
                    summary_paths = methylseg.run_on_all_chroms(
                        sample_info=sample_info,
                        chroms=all_chroms,
                        min_probes=5,
                    )
        except Exception as e:
            self._write_log(f"Error processing MethylSeg chromosomes: {e}")

        finally:
            sampler.stop()
            peak_rss = max(peak_rss, sampler.peak_rss)

        end = time.perf_counter()

        runtime = None if cache_used else (end - start)
        peak_mem_mb = None if cache_used else (peak_rss / 1e6)

        if runtime is not None:
            self._write_log(
                f"MethylSeg run | time={runtime:.2f}s | peak_mem={peak_mem_mb:.1f} MB"
            )
        tool_name = f"MethylSeg_{tool_suffix}" if tool_suffix else "MethylSeg"
        self.run_stats.append(
            RunStats(
                tool_name=tool_name,
                runtime_sec=runtime,
                mem_peak_mb=peak_mem_mb,
            )
        )
        return segmented_sample_info, summary_paths

    def _collect_methylseg_output_paths(
        self,
        output_dir: Path,
        sample_info: SampleInfo,
    ) -> list[str]:
        output_dir = Path(output_dir)
        summary_file_dir = output_dir / "summary_files"
        chroms = (
            sample_info.meth_data["CpG_chrm"].dropna().astype(str).unique().tolist()
        )
        output_paths: list[str] = []

        for state in MethylationStates:
            output_paths.append(
                str(summary_file_dir / f"segments_raw_{state.name}.bed")
            )
            output_paths.append(
                str(summary_file_dir / f"segments_cleaned_{state.name}.bed")
            )
            for chrom in chroms:
                output_paths.append(
                    str(
                        output_dir
                        / "clean_regions"
                        / f"segments_cleaned_{chrom}_{sample_info.sample_id}_{state.name}.bed"
                    )
                )

        return output_paths

    def run_tool(
        self,
    ) -> dict:
        prep_dir = self._prep_dir()
        hm450_meth_data = prep_dir / "450k.beta"
        hm450_sample_info, hm450_removed_cpgs = (
            MethylSegToolPathway.prepare_sample_info(
                sample_name=self.sample_id + "_hm450k",
                sample_file=hm450_meth_data,
                resolution="450k",
                remove_low_coverage_like_cpgs=True,
            )
        )
        self.hm450_removed_cpgs = self._annotate_removed_cpg_reason(
            hm450_removed_cpgs,
            "low coverage like",
        )
        self._write_removed_cpgs("hm450k", self.hm450_removed_cpgs)
        methyl_seg_out_dir = Path(self.out_dir) / self.sample_id / "out" / "hm450k"
        methyl_seg_out_dir.mkdir(parents=True, exist_ok=True)

        if self.train_sample_info is not None:
            hm450k_train_sample = self.train_sample_info
        else:
            hm450k_train_sample = hm450_sample_info

        self._write_log(
            f"Training MethylSeg on sample {hm450k_train_sample.sample_id} with window specs {self.hm450_window_specs} and HMM type {self.hm450_hmm_type} and params {self.hm450_hmm_params}"
        )

        hm450_methylseg = MethylSegToolPathway(
            train_sample_info=hm450k_train_sample,
            window_specs=self.hm450_window_specs,
            n_states=self.n_states,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            high_cutoff=self.high_cutoff,
            out_dir=methyl_seg_out_dir,
            random_state=self.random_state,
            hmm_type=self.hm450_hmm_type,
            hmm_params=self.hm450_hmm_params,
            min_region_length=self.methylseg_clean_min_region_length,
            min_region_cpgs=self.methylseg_clean_min_cpgs,
            merge_gap_bp=self.methylseg_clean_merge_gap_bp,
            state_assignment_method=self.state_assignment_method,
        )

        _, summary_files = self._run_methylseg_on_each_chromosome(
            hm450_methylseg, hm450_sample_info, tool_suffix="HM450K"
        )

        methyl_seg_out_dir = Path(self.out_dir) / self.sample_id / "out" / "wgbs"
        methyl_seg_out_dir.mkdir(parents=True, exist_ok=True)
        wgbs_meth_data = prep_dir / "wgbs.tsv"
        wgbs_sample_info, wgbs_removed_cpgs = MethylSegToolPathway.prepare_sample_info(
            sample_name=self.sample_id + "_wgbs",
            sample_file=wgbs_meth_data,
            resolution="wgbs",
            min_coverage=self.min_coverage,
            remove_low_coverage_like_cpgs=True,
        )
        self.wgbs_removed_cpgs = self._annotate_wgbs_removed_cpg_reasons(
            wgbs_removed_cpgs
        )
        self._write_removed_cpgs("wgbs", self.wgbs_removed_cpgs)

        if self.train_sample_info is not None:
            wgbs_train_sample = self.train_sample_info
        else:
            wgbs_train_sample = wgbs_sample_info

        self._write_log(
            f"Training MethylSeg on sample {wgbs_train_sample.sample_id} with window specs {self.wgbs_window_specs} and HMM type {self.wgbs_hmm_type} and params {self.wgbs_hmm_params}"
        )

        wgbs_methylseg = MethylSegToolPathway(
            train_sample_info=wgbs_train_sample,
            window_specs=self.wgbs_window_specs,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            high_cutoff=self.high_cutoff,
            n_states=self.n_states,
            out_dir=methyl_seg_out_dir,
            random_state=self.random_state,
            hmm_type=self.wgbs_hmm_type,
            hmm_params=self.wgbs_hmm_params,
            min_region_length=self.methylseg_clean_min_region_length,
            min_region_cpgs=self.methylseg_clean_min_cpgs,
            merge_gap_bp=self.methylseg_clean_merge_gap_bp,
            state_assignment_method=self.state_assignment_method,
        )

        _, wgbs_summary_files = self._run_methylseg_on_each_chromosome(
            wgbs_methylseg, wgbs_sample_info, tool_suffix="WGBS"
        )
        summary_files += wgbs_summary_files
        self._write_prep_config(
            {
                "tool_name": "MethylSeg",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": prep_dir / "wgbs.tsv",
                    "wgbs_beta": prep_dir / "wgbs.beta",
                    "wgbs_meth_ref": prep_dir / "wgbs_meth_ref.tsv",
                    "wgbs_450k_intersect_tsv": prep_dir / "wgbs_450k_intersect.tsv",
                    "hm450k_beta": prep_dir / "450k.beta",
                    "hm450k_meth_ref": prep_dir / "450k_meth_ref.tsv",
                },
                "tool_runtime_settings": {
                    "fit_methyl_seg": self.fit_methyl_seg,
                    "methyl_seg_config_path": self.methyl_seg_config,
                    "train_sample": self.train_sample,
                    "train_chrom": self.train_chrom,
                    "random_state": self.random_state,
                },
                "platforms": {
                    "wgbs": {
                        "min_coverage": self.min_coverage,
                        "remove_low_coverage_like_cpgs": True,
                        "window_specs": self.wgbs_window_specs,
                        "n_states": self.n_states,
                        "int_low_cutoff": self.int_low_cutoff,
                        "int_high_cutoff": self.int_high_cutoff,
                        "high_cutoff": self.high_cutoff,
                        "hmm_type": self.wgbs_hmm_type,
                        "hmm_params": self.wgbs_hmm_params,
                        "min_region_length": self.methylseg_clean_min_region_length,
                        "min_region_cpgs": self.methylseg_clean_min_cpgs,
                        "merge_gap_bp": self.methylseg_clean_merge_gap_bp,
                        "state_assignment_method": self.state_assignment_method,
                    },
                    "hm450k": {
                        "min_coverage": None,
                        "remove_low_coverage_like_cpgs": True,
                        "window_specs": self.hm450_window_specs,
                        "n_states": self.n_states,
                        "int_low_cutoff": self.int_low_cutoff,
                        "int_high_cutoff": self.int_high_cutoff,
                        "high_cutoff": self.high_cutoff,
                        "hmm_type": self.hm450_hmm_type,
                        "hmm_params": self.hm450_hmm_params,
                        "min_region_length": self.methylseg_clean_min_region_length,
                        "min_region_cpgs": self.methylseg_clean_min_cpgs,
                        "merge_gap_bp": self.methylseg_clean_merge_gap_bp,
                        "state_assignment_method": self.state_assignment_method,
                    },
                },
            }
        )
        return {
            "summary_bed_files": summary_files,
        }


class MethylLassoPathway(MethylToolPathway):

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )

    def run_data_prep(
        self,
    ) -> dict:
        out_dir = Path(self.out_dir)
        sample_dir = out_dir / self.sample_id
        prep_dir = sample_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)

        shared_prep = self._resolve_shared_prep()
        if shared_prep is not None:
            wgbs_tsv = self._materialize_shared_file(
                shared_prep.wgbs_tsv,
                prep_dir / "wgbs.tsv",
            )
        else:
            wgbs_tsv = self._create_readable_meth_file(prep_dir / "wgbs.tsv")

        methylasso_meth_file = prep_dir / "methylasso_input.tsv"
        if not methylasso_meth_file.exists() or self.force_recreate:
            df = pd.read_csv(
                wgbs_tsv,
                sep="	",
                header=None,
                names=["chrom", "start", "end", "meth", "coverage"],
            )
            df["meth_percent"] = df["meth"] / df["coverage"]

            df.dropna(subset=["meth_percent", "coverage"], how="any", inplace=True)

            df[["chrom", "start", "end", "meth_percent", "coverage"]].to_csv(
                methylasso_meth_file, sep="	", index=False, header=None
            )

        return {
            "methylasso_meth_file": methylasso_meth_file,
        }

    def run_tool(
        self,
    ) -> dict:
        out_dir = Path(self.out_dir)

        sample_dir = out_dir / self.sample_id
        input_file = sample_dir / "prep" / f"methylasso_input.tsv"
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "MethyLasso",
            "methylasso",
            "--n1",
            self.sample_id,
            "--c1",
            str(input_file.resolve()),
            "--cov",
            "5",
            "--meth",
            "4",
            "-q",
            "0.05",
        ]

        cwd = sample_dir / "out"
        start = time.perf_counter()
        meth_lasso_std_out = (
            sample_dir / "logs" / "job_logs" / f"methylasso_{start:.0f}.stdout"
        )
        meth_lasso_std_err = (
            sample_dir / "logs" / "job_logs" / f"methylasso_{start:.0f}.stderr"
        )
        cwd.mkdir(parents=True, exist_ok=True)
        methylasso_run_stats = self._run(
            cmd,
            cwd=str(cwd.resolve()),
            stdout_path=meth_lasso_std_out,
            stderr_path=meth_lasso_std_err,
        )

        pmd_tsv = cwd / f"{self.sample_id}_pmd.tsv"
        lmr_umr_dmv_tsv = cwd / f"{self.sample_id}_lmr_umr_dmv.tsv"
        self.run_stats.append(RunStats(tool_name="methylasso", **methylasso_run_stats))
        self._write_prep_config(
            {
                "tool_name": "MethylLasso",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": self._prep_dir() / "wgbs.tsv",
                    "methylasso_input_tsv": input_file,
                },
                "prep_filters": {
                    "wgbs": [
                        "compute meth_percent = meth / coverage",
                        "drop rows where meth_percent is NA",
                        "drop rows where coverage is NA",
                    ],
                },
                "tool_commands": {
                    "methylasso": cmd,
                },
                "tool_runtime_settings": {
                    "cov_cutoff": 5,
                    "meth_cutoff": 4,
                    "q_value_cutoff": 0.05,
                },
            }
        )

        return {
            "pmd_tsv": str(pmd_tsv),
            "lmr_umr_dmv_tsv": str(lmr_umr_dmv_tsv),
        }


class MMSeekRPathway(MethylToolPathway):

    def __init__(
        self,
        sample_id,
        meth_file,
        genome,
        out_dir,
        force_recreate=False,
        print_logs: bool = True,
        shared_prep: Any = None,
    ):
        super().__init__(
            sample_id,
            meth_file,
            genome,
            out_dir,
            force_recreate,
            print_logs,
            shared_prep,
        )

    def _wgbs_to_mmseekr_format(self, wgbs_tsv: Path, out_meth: Path) -> Path:
        """
        Convert a WGBS table (chrom, start, end, meth, coverage) to MMSeekR
        text input:
        chr  pos  T  M

        MMSeekR joins against NNscore tables by 1-based CpG position, so BED
        starts must be shifted by +1 before writing pos.
        """
        if out_meth.exists() and not self.force_recreate:
            return out_meth
        out_meth.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(
            wgbs_tsv,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "meth", "coverage"],
        )
        # normalize chromosome names → chrN
        df["chrom"] = df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)

        # keep autosomes only
        valid_chroms = {str(i) for i in range(1, 23)}
        df = df[df["chrom"].isin(valid_chroms)]

        df["chrom"] = "chr" + df["chrom"]
        df["pos"] = df["start"].astype(int) + 1
        df["M"] = df["meth"].astype(int)
        df["T"] = df["coverage"].astype(int)

        mmseekr_df = df[
            [
                "chrom",
                "pos",
                "T",
                "M",
            ]
        ]
        # mmseekr_df = mmseekr_df[mmseekr_df["T"] >= 5]

        # write back in MMSeekR text format
        mmseekr_df.to_csv(
            out_meth,
            sep="\t",
            header=None,
            index=False,
        )
        return out_meth

    def run_data_prep(
        self,
    ) -> dict:
        prep_dir = self._prep_dir()

        shared_prep = self._resolve_shared_prep()
        if shared_prep is not None:
            wgbs_tsv = self._materialize_shared_file(
                shared_prep.wgbs_tsv,
                prep_dir / "wgbs.tsv",
            )
        else:
            wgbs_tsv = self._create_readable_meth_file(prep_dir / "wgbs.tsv")

        mmseekr_meth = self._wgbs_to_mmseekr_format(
            wgbs_tsv,
            prep_dir / "mmseekr_input.tsv",
        )

        return {
            "mmseekr_meth": mmseekr_meth,
        }

    def run_tool(
        self,
    ) -> dict:
        r_script = FILES / Path("run_mmseekr.r")
        if not r_script.exists():
            raise FileNotFoundError(f"Missing R script: {r_script}")

        cmd = [
            "Rscript",
            str(r_script),
            "--genome",
            self.genome,
            "--sample",
            str(self.sample_id),
            "--out_dir",
            str(self.out_dir),
            "--trainChr",
            "chr1",
            "--nCGbin",
            "201",
            "--num_cores",
            "6",
        ]
        out_bed = (
            Path(self.out_dir)
            / Path(self.sample_id)
            / Path("out")
            / Path(f"{self.sample_id}.multiModel.PMDs.bed")
        )
        out_bed.parent.mkdir(parents=True, exist_ok=True)
        log_file = (
            Path(self.out_dir)
            / Path(self.sample_id)
            / Path("logs")
            / Path("job_logs")
            / Path("mmseekr.log")
        )
        mmseekr_run_stats = {}
        if not out_bed.exists() or self.force_recreate:
            mmseekr_run_stats = self._run(cmd)
            if not log_file.exists():
                raise FileNotFoundError(f"Missing mmseekr log file: {log_file}")
            if self.print_logs:
                for line in open(log_file):
                    print(line.strip())
        self.run_stats.append(RunStats(tool_name="mmseekr", **mmseekr_run_stats))
        self._write_prep_config(
            {
                "tool_name": "MMSeekR",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": self._prep_dir() / "wgbs.tsv",
                    "mmseekr_input_tsv": self._prep_dir() / "mmseekr_input.tsv",
                },
                "prep_filters": {
                    "wgbs": [
                        "strip leading chr prefix before autosome test",
                        "keep chromosomes 1-22 only",
                        "convert CpG_beg to 1-based pos",
                    ],
                },
                "tool_runtime_settings": {
                    "script_path": r_script,
                    "trainChr": "chr1",
                    "nCGbin": 201,
                    "num_cores": 6,
                },
                "tool_commands": {
                    "run_mmseekr": cmd,
                },
            }
        )
        return {
            "pmd_bed": str(out_bed),
            "log_file": str(log_file),
        }


def _run_pathway_job(
    tool_name: str,
    pathway: MethylToolPathway,
    shared_outputs: SharedPrepArtifacts,
) -> tuple[str, Any]:
    pathway.shared_prep = shared_outputs
    pathway_results = pathway.run()
    return tool_name, pathway_results


class MethylToolComparator:
    def __init__(
        self,
        config_file: Path,
        out_dir: Path,
        force_recreate: bool = False,
        n_jobs: int = 10,
        alt_params={},
    ):
        with open(config_file) as fh:
            config = yaml.safe_load(fh)
        sample_id = config["sample"]
        meth_file = config["meth_file"]
        genome = config["genome"]
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        comparison_root = Path(out_dir) / "comparison"
        self.out_dir: Path = comparison_root / sample_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sample_id = sample_id
        self.force_recreate = force_recreate
        self.n_jobs = n_jobs
        self.shared_prep_manager = SharedPrepManager(
            sample_id,
            meth_file,
            genome,
            comparison_root,
            force_recreate=force_recreate,
        )
        self.pathways = {
            "DNMTools": DNMToolsPathway(
                sample_id,
                meth_file,
                genome,
                Path(out_dir) / "dnmtools",
                force_recreate=force_recreate,
                shared_prep=self.shared_prep_manager,
            ),
            "MethylSeekR": MethylseekRPathway(
                sample_id,
                meth_file,
                genome,
                Path(out_dir) / "methylseekr",
                force_recreate=force_recreate,
                shared_prep=self.shared_prep_manager,
            ),
            "MethylLasso": MethylLassoPathway(
                sample_id,
                meth_file,
                genome,
                Path(out_dir) / "methyl_lasso",
                force_recreate=force_recreate,
                shared_prep=self.shared_prep_manager,
            ),
            "MMSeekR": MMSeekRPathway(
                sample_id,
                meth_file,
                genome,
                Path(out_dir) / "mmseekr",
                force_recreate=force_recreate,
                shared_prep=self.shared_prep_manager,
            ),
            "MethylSeg": MethylSegPathway(
                sample_id,
                meth_file,
                genome,
                Path(out_dir) / "methylseg",
                force_recreate=force_recreate,
                shared_prep=self.shared_prep_manager,
                train_sample=None,
                train_sample_file=None,
                **alt_params.get("MethylSeg", {}),
            ),
        }
        self._plot_universe_cache: dict[str, pd.DataFrame] = {}

    def run_all(self):
        shared_outputs = self.shared_prep_manager.prepare()

        for pathway in self.pathways.values():
            pathway.shared_prep = shared_outputs

        results = {}
        if self.n_jobs == 1 or len(self.pathways) == 1:
            for tool_name, pathway in self.pathways.items():
                pathway_results = pathway.run()
                results[tool_name] = pathway_results
            return results

        max_workers = min(self.n_jobs, len(self.pathways))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_pathway_job, tool_name, pathway, shared_outputs)
                for tool_name, pathway in self.pathways.items()
            ]
            for future in as_completed(futures):
                tool_name, pathway_results = future.result()
                results[tool_name] = pathway_results

        return {tool_name: results[tool_name] for tool_name in self.pathways}

    def _load_plot_universe(self, platform: str) -> pd.DataFrame:
        if platform in self._plot_universe_cache:
            return self._plot_universe_cache[platform]

        prep_dir = Path(self.pathways["MethylSeg"].out_dir) / self.sample_id / "prep"
        if platform == "wgbs":
            raw_path = prep_dir / "wgbs.tsv"
            if raw_path.exists():
                tracking_df = MethylToolPathway._load_wgbs_tracking_df(raw_path)
            else:
                tracking_df = pd.DataFrame(
                    columns=["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
                )
        elif platform == "hm450k":
            raw_path = prep_dir / "wgbs_450k_intersect.tsv"
            if raw_path.exists():
                tracking_df = MethylToolPathway._load_intersect_tracking_df(raw_path)
            else:
                tracking_df = pd.DataFrame(
                    columns=["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
                )
        else:
            raise ValueError(f"Unsupported plot platform: {platform}")

        plot_df = MethylToolPathway._build_plot_universe_df(tracking_df)
        self._plot_universe_cache[platform] = plot_df
        return plot_df

    @staticmethod
    def _normalize_removed_cpgs_for_plot(
        cpg_df: pd.DataFrame,
        *,
        ch_col: str = "CpG_chrm",
        x_col: str = "CpG_beg",
        end_col: str = "CpG_end",
        reason_col: str = "removed_reason",
        default_reason: str = "",
    ) -> pd.DataFrame:
        if cpg_df is None or cpg_df.empty:
            return pd.DataFrame(columns=[ch_col, x_col, end_col, reason_col])

        required_cols = [ch_col, x_col, end_col]
        if not set(required_cols).issubset(cpg_df.columns):
            return pd.DataFrame(columns=[ch_col, x_col, end_col, reason_col])

        plot_df = cpg_df.loc[:, required_cols].copy()
        plot_df[ch_col] = plot_df[ch_col].astype(str)
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce").astype("Int64")
        plot_df[end_col] = pd.to_numeric(plot_df[end_col], errors="coerce").astype(
            "Int64"
        )
        if reason_col in cpg_df.columns:
            plot_df[reason_col] = cpg_df[reason_col].fillna(default_reason).astype(str)
        else:
            plot_df[reason_col] = default_reason
        plot_df = plot_df.dropna(subset=[x_col, end_col]).reset_index(drop=True)
        if plot_df.empty:
            return pd.DataFrame(columns=[ch_col, x_col, end_col, reason_col])
        plot_df[x_col] = plot_df[x_col].astype(np.int64)
        plot_df[end_col] = plot_df[end_col].astype(np.int64)
        return plot_df.drop_duplicates(required_cols, keep="first").reset_index(
            drop=True
        )

    @classmethod
    def _build_reverse_intersection_removed_cpgs(
        cls,
        source_df: pd.DataFrame,
        retained_df: pd.DataFrame,
        *,
        ch_col: str = "CpG_chrm",
        x_col: str = "CpG_beg",
        end_col: str = "CpG_end",
        reason: str,
    ) -> pd.DataFrame:
        source_coords = cls._normalize_removed_cpgs_for_plot(
            source_df,
            ch_col=ch_col,
            x_col=x_col,
            end_col=end_col,
            default_reason=reason,
        )
        retained_coords = cls._normalize_removed_cpgs_for_plot(
            retained_df,
            ch_col=ch_col,
            x_col=x_col,
            end_col=end_col,
        )
        if source_coords.empty:
            return source_coords
        if retained_coords.empty:
            return source_coords

        retained_index = pd.MultiIndex.from_frame(
            retained_coords.loc[:, [ch_col, x_col, end_col]]
        )
        source_index = pd.MultiIndex.from_frame(
            source_coords.loc[:, [ch_col, x_col, end_col]]
        )
        return source_coords.loc[~source_index.isin(retained_index)].reset_index(
            drop=True
        )

    @classmethod
    def _match_removed_cpgs_to_plot(
        cls,
        coord_index: pd.MultiIndex,
        cpg_df: pd.DataFrame,
        *,
        ch_col: str = "CpG_chrm",
        x_col: str = "CpG_beg",
        end_col: str = "CpG_end",
        reason_col: str = "removed_reason",
        default_reason: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_df = cls._normalize_removed_cpgs_for_plot(
            cpg_df,
            ch_col=ch_col,
            x_col=x_col,
            end_col=end_col,
            reason_col=reason_col,
            default_reason=default_reason,
        )
        mask = np.zeros(len(coord_index), dtype=bool)
        reasons = np.full(len(coord_index), "", dtype=object)
        if normalized_df.empty:
            return mask, reasons

        lookup = normalized_df.set_index([ch_col, x_col, end_col])[reason_col]
        mask = coord_index.isin(lookup.index)
        if mask.any():
            reasons[mask] = (
                lookup.reindex(coord_index[mask]).fillna(default_reason).to_numpy()
            )
        return mask, reasons

    def _ensure_3cols(self, df):
        # If first three columns exist but no named columns, set names
        if df.shape[1] >= 3 and df.columns.tolist() == [0, 1, 2]:
            df = df.rename(columns={0: "chr", 1: "start", 2: "end"})
        # If start/end might be floats / strings, cast to ints
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        # Keep only chr,start,end (plus any other columns optionally)
        return df

    def _load_regions(self):
        if hasattr(self, "region_dfs") and hasattr(self, "region_beds"):
            return

        methylseekr = self.pathways["MethylSeekR"]
        dnmtools = self.pathways["DNMTools"]
        methylseg = self.pathways["MethylSeg"]
        methylasso = self.pathways["MethylLasso"]
        mmseekr = self.pathways["MMSeekR"]

        methylseekr_regions = pd.read_csv(
            Path(methylseekr.out_dir)
            / self.sample_id
            / "out"
            / f"methylseekr_PMDs.bed",
            sep="\t",
        )

        dnm_tools_regions = pd.read_csv(
            Path(dnmtools.out_dir) / self.sample_id / "out" / f"dnmtools_PMDs.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label", "score", "strand"],
        )

        arraymode_dnm_tools_regions = pd.read_csv(
            Path(dnmtools.out_dir)
            / self.sample_id
            / "out"
            / f"arraymode.dnmtools_PMDs.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label", "score", "strand"],
        )

        pmr_dnm_tools_regions = pd.read_csv(
            Path(dnmtools.out_dir) / self.sample_id / "out" / f"pmr.dnmtools_PMDs.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label", "score", "strand"],
        )

        methylasso_regions = pd.read_csv(
            Path(methylasso.out_dir)
            / self.sample_id
            / "out"
            / f"{self.sample_id}_pmd.tsv",
            sep="\t",
            header=0,
            names=["chr", "start", "end", "num.cpgs", "meth", "std", "category"],
        )

        mmseekr_regions = pd.read_csv(
            Path(mmseekr.out_dir)
            / self.sample_id
            / "out"
            / f"{self.sample_id}.multiModel.PMDs.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end"],
        )

        methyl_seg_hm450k_regions = pd.read_csv(
            Path(methylseg.out_dir)
            / self.sample_id
            / "out"
            / "hm450k"
            / "summary_files"
            / "segments_cleaned_PMD.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label"],
        )

        methyl_seg_wgbs_regions = pd.read_csv(
            Path(methylseg.out_dir)
            / self.sample_id
            / "out"
            / "wgbs"
            / "summary_files"
            / "segments_cleaned_PMD.bed",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label"],
        )

        methylseekr_df = self._ensure_3cols(methylseekr_regions)
        dnm_tools_df = self._ensure_3cols(dnm_tools_regions)
        methylseg_hm450k_df = self._ensure_3cols(methyl_seg_hm450k_regions)
        methylseg_wgbs_df = self._ensure_3cols(methyl_seg_wgbs_regions)
        array_dnm_df = self._ensure_3cols(arraymode_dnm_tools_regions)
        pmr_dnm_df = self._ensure_3cols(pmr_dnm_tools_regions)
        methylasso_df = self._ensure_3cols(methylasso_regions)
        mmseekr_df = self._ensure_3cols(mmseekr_regions)

        # Convert to BedTool objects
        bt_methylseekr = BedTool.from_dataframe(
            methylseekr_df.loc[:, ["chr", "start", "end"]]
        )
        bt_dnmtools = BedTool.from_dataframe(
            dnm_tools_df.loc[:, ["chr", "start", "end"]]
        )
        bt_methylseg_hm450k = BedTool.from_dataframe(
            methylseg_hm450k_df.loc[:, ["chr", "start", "end"]]
        )
        bt_methylseg_wgbs = BedTool.from_dataframe(
            methylseg_wgbs_df.loc[:, ["chr", "start", "end"]]
        )
        bt_dnmtools_array = BedTool.from_dataframe(
            array_dnm_df.loc[:, ["chr", "start", "end"]]
        )
        bt_dnmtools_pmr = BedTool.from_dataframe(
            pmr_dnm_df.loc[:, ["chr", "start", "end"]]
        )
        bt_methylasso = BedTool.from_dataframe(
            methylasso_df.loc[:, ["chr", "start", "end"]]
        )
        bt_mmseekr = BedTool.from_dataframe(mmseekr_df.loc[:, ["chr", "start", "end"]])

        self.region_dfs = {
            "methylseekr": methylseekr_df,
            "dnmtools": dnm_tools_df,
            "methylseg_hm450k": methylseg_hm450k_df,
            "methylseg_wgbs": methylseg_wgbs_df,
            "dnmtools_array": array_dnm_df,
            "dnmtools_pmr": pmr_dnm_df,
            "methylasso": methylasso_df,
            "mmseekr": mmseekr_df,
        }
        self.region_beds = {
            "methylseekr": bt_methylseekr,
            "dnmtools": bt_dnmtools,
            "methylseg_hm450k": bt_methylseg_hm450k,
            "methylseg_wgbs": bt_methylseg_wgbs,
            "dnmtools_array": bt_dnmtools_array,
            "dnmtools_pmr": bt_dnmtools_pmr,
            "methylasso": bt_methylasso,
            "mmseekr": bt_mmseekr,
        }

    def _parse_genome_coverage_pct(self, bt: BedTool, genome: str) -> float:
        """
        Parse genome coverage percentage from pybedtools genome_coverage(hist=True).

        Returns:
            Percent of genome covered by >=1 bp.
        """
        gcov = bt.genome_coverage(genome=genome)

        for line in gcov:
            fields = line.fields
            if fields[0] == "genome":
                depth = int(fields[1])
                if depth >= 1:
                    return float(fields[4]) * 100.0

        return 0.0

    def _merged_lengths(self, bt: BedTool) -> np.ndarray:
        """
        Merge intervals and return an array of region lengths (bp).
        """
        merged = bt.merge()
        return np.array([iv.end - iv.start for iv in merged], dtype=float)

    def _get_stats(self) -> pd.DataFrame:
        """
        Compute summary statistics for each tool's regions.

        Metrics:
        - n_regions
        - total_bp
        - genome_coverage_pct
        - mean / median / std / min / max region length
        """
        stats = {}

        # Assume all tools use the same genome
        genome = self.pathways["MethylSeg"].genome

        for name, bt in self.region_beds.items():
            # Merge regions to avoid double-counting
            lengths = self._merged_lengths(bt)

            n_regions = len(lengths)
            total_bp = lengths.sum() if n_regions > 0 else 0.0

            genome_cov_pct = self._parse_genome_coverage_pct(bt, genome)

            stats[name] = {
                "n_regions": int(n_regions),
                "total_bp": int(total_bp),
                "genome_coverage_pct": genome_cov_pct,
                "mean_length": lengths.mean() if n_regions > 0 else np.nan,
                "median_length": np.median(lengths) if n_regions > 0 else np.nan,
                "std_length": lengths.std(ddof=1) if n_regions > 1 else np.nan,
                "min_length": lengths.min() if n_regions > 0 else np.nan,
                "max_length": lengths.max() if n_regions > 0 else np.nan,
            }

        return pd.DataFrame.from_dict(stats, orient="index")

    def _total_bp(self, bed):
        bt = pbt.BedTool(bed) if not isinstance(bed, pbt.BedTool) else bed
        merged = bt.merge()
        total = 0
        for iv in merged:
            total += int(iv.end) - int(iv.start)
        return total

    def _intersection_bp(self, a, b):
        a_bt = pbt.BedTool(a) if not isinstance(a, pbt.BedTool) else a
        it = a_bt.intersect(b, wao=True)
        total_overlap = 0
        for line in it:
            vals = line.fields
            try:
                overlap = int(vals[-1])
            except Exception:
                overlap = 0
            total_overlap += overlap
        return total_overlap

    def _compute_matrices(self, beds: Dict[str, Any]):
        names = list(beds.keys())
        pct_cover = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        jaccard = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        totals = {}
        for nm in names:
            totals[nm] = self._total_bp(beds[nm])
            if totals[nm] == 0:
                print(
                    f"Warning: tool {nm} has zero total bp after merging; results may be undefined."
                )
        for a in names:
            for b in names:
                inter = self._intersection_bp(beds[a], beds[b])
                pct = (inter / totals[a]) if totals[a] > 0 else np.nan
                pct_cover.loc[a, b] = pct
                union = totals[a] + totals[b] - inter
                j = (inter / union) if union > 0 else np.nan
                jaccard.loc[a, b] = j
        return pct_cover, jaccard

    def _create_heatmaps(self):
        heatmatp_file = Path(self.out_dir) / "region_heatmaps.html"

        if heatmatp_file.exists() and not self.force_recreate:
            return
        pct_cover_df, jaccard_df = self._compute_matrices(self.region_beds)

        #TODO: save jaccard_df and pct_cover_df to CSV for later inspection
        jaccard_df.to_csv(Path(self.out_dir) / "jaccard_matrix.csv")
        pct_cover_df.to_csv(Path(self.out_dir) / "pct_cover_matrix.csv")

        names = list(self.region_beds.keys())
        pct_pct_df = (pct_cover_df * 100).round(4)
        pct_not_df = (100.0 - pct_pct_df).round(4)
        jaccard_df = jaccard_df.round(4)

        white_red = [
            [0.0, "white"],
            [1.0, "red"],
        ]

        fig = make_subplots(
            rows=3,
            cols=1,
            # shared_xaxes=True,
            # vertical_spacing=0.04,
            subplot_titles=(
                "Percent of A covered by B (%)",
                "Percent of A NOT covered by B (%)",
                "Jaccard index (intersection / union)",
            ),
        )

        # Heatmap 1: percent covered 0-100
        fig.add_trace(
            go.Heatmap(
                z=pct_pct_df.values,
                x=names,
                y=names,
                zmin=0,
                zmax=100,
                colorscale=white_red,
                colorbar=dict(title="% covered", lenmode="fraction", len=0.25, y=0.85),
                hovertemplate="A: %{y}<br>B: %{x}<br>% covered: %{z:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Heatmap 2: percent not covered 0-100
        fig.add_trace(
            go.Heatmap(
                z=pct_not_df.values,
                x=names,
                y=names,
                zmin=0,
                zmax=100,
                colorscale=white_red,
                colorbar=dict(
                    title="% not covered", lenmode="fraction", len=0.25, y=0.52
                ),
                hovertemplate="A: %{y}<br>B: %{x}<br>% not covered: %{z:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Heatmap 3: jaccard 0-1
        fig.add_trace(
            go.Heatmap(
                z=jaccard_df.values,
                x=names,
                y=names,
                zmin=0,
                zmax=1,
                colorscale=white_red,
                colorbar=dict(title="Jaccard", lenmode="fraction", len=0.25, y=0.19),
                hovertemplate="A: %{y}<br>B: %{x}<br>Jaccard: %{z:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=1200,
            width=1000,
            title_text="Pairwise bed comparisons",
            template="plotly_white",
            margin=dict(l=120, r=40, t=100, b=80),
        )

        # ensure y-axis displays in natural order (top=A[0])
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickangle=45)

        fig.write_html(
            heatmatp_file,
            include_plotlyjs="cdn",
        )

    def _states_from_bed_for_positions(
        self,
        chrom_positions: np.ndarray,
        intervals: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Given a sorted numpy array of positions on one chromosome and a list of
        (start,end) intervals (not necessarily sorted), return a 0/1 numpy array
        marking each position that lies inside any interval.
        """
        if len(intervals) == 0:
            return np.zeros_like(chrom_positions, dtype=int)

        # sort and merge intervals
        intervals_sorted = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = []
        cur_s, cur_e = intervals_sorted[0]
        for s, e in intervals_sorted[1:]:
            if s <= cur_e:  # overlap or adjacent
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        pos = chrom_positions
        mask = np.zeros(len(pos), dtype=bool)
        for s, e in merged:
            i = np.searchsorted(pos, s, side="left")
            j = np.searchsorted(pos, e, side="right")
            if j > i:
                mask[i:j] = True
        return mask.astype(int)

    def _get_intervals_from_bed_like(
        self, bed_like: Any, chrom: str
    ) -> list[tuple[int, int]]:
        """
        Accepts a pybedtools.BedTool, pandas.DataFrame, or iterable of tuples and
        returns a list of (start,end) intervals for the requested chromosome.
        """
        intervals = []

        # pybedtools BedTool (has .fn attribute and is iterable)
        if hasattr(bed_like, "__iter__") and hasattr(bed_like, "fn"):
            for iv in bed_like:
                if iv.chrom == chrom:
                    intervals.append((int(iv.start), int(iv.end)))
            return intervals

        # pandas DataFrame with chr,start,end
        if isinstance(bed_like, pd.DataFrame):
            cols = bed_like.columns.tolist()
            if set(["chr", "start", "end"]).issubset(cols):
                df_chr = bed_like[bed_like["chr"] == chrom]
                return [
                    (int(s), int(e))
                    for s, e in df_chr[["start", "end"]].itertuples(index=False)
                ]
            else:
                # assume first three columns: chr,start,end
                df_chr = bed_like[bed_like.iloc[:, 0] == chrom]
                return [
                    (int(s), int(e))
                    for s, e in df_chr.iloc[:, 1:3].itertuples(index=False)
                ]

        # generic iterable of tuples (chr,start,end)
        try:
            for t in bed_like:
                if t[0] == chrom:
                    intervals.append((int(t[1]), int(t[2])))
            return intervals
        except Exception:
            return []

    def _plot_chromosome_tools(
        self,
        meth_df: pd.DataFrame,
        chrom: str,
        tools: Sequence[str],
        beds_dict: Dict[str, Any],
        x_col: str = "CpG_beg",
        end_col: str = "CpG_end",
        y_col: str = "beta",
        ch_col: str = "CpG_chrm",
        sample_name: Optional[str] = None,
        max_points: int = 120_000,
        point_size: int = 3,
        html_out: Optional[str] = None,
        removed_cpgs_by_tool: Optional[Dict[str, pd.DataFrame]] = None,
        downsampled_cpgs_by_tool: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Produce vertically stacked interactive plots (one per tool),
        sharing the same x-axis (chromosome-wide zoomable view).

        - tools: sequence of tool names (keys in beds_dict). Can be any length >=1.
        - beds_dict: mapping tool name -> bed-like intervals (pybedtools BedTool, DataFrame, or iterable)
        Returns (fig, df_chr) where df_chr is the full chromosome CpG DataFrame (not downsampled).
        """
        n_tools = len(tools)
        if n_tools < 1:
            raise ValueError("Provide at least one tool name in `tools`.")

        # Filter to chromosome
        df_chr = meth_df[meth_df[ch_col] == chrom].copy().reset_index(drop=True)
        if df_chr.empty:
            if df_chr.empty:
                msg = f"[INFO] No CpGs found for {sample_name or 'sample'} {chrom}, skipping."
                print(msg)
                return None, df_chr

        # sort by genomic position
        df_chr = df_chr.sort_values(x_col).reset_index(drop=True)
        positions = df_chr[x_col].to_numpy(dtype=int)
        coord_index = pd.MultiIndex.from_frame(df_chr[[ch_col, x_col, end_col]])

        # compute states (0/1) for each tool
        states_per_tool = {}
        removed_mask_per_tool = {}
        removed_reason_per_tool = {}
        downsampled_mask_per_tool = {}
        downsampled_reason_per_tool = {}
        for t in tools:
            if t not in beds_dict:
                raise ValueError(
                    f"tool '{t}' not in beds_dict keys: {list(beds_dict.keys())}"
                )
            intervals = self._get_intervals_from_bed_like(beds_dict[t], chrom)
            states_per_tool[t] = self._states_from_bed_for_positions(
                positions, intervals
            )

            removed_df = (
                None if removed_cpgs_by_tool is None else removed_cpgs_by_tool.get(t)
            )
            removed_mask_per_tool[t], removed_reason_per_tool[t] = (
                self._match_removed_cpgs_to_plot(
                    coord_index,
                    removed_df,
                    ch_col=ch_col,
                    x_col=x_col,
                    end_col=end_col,
                    default_reason="removed",
                )
            )

            downsampled_df = (
                None
                if downsampled_cpgs_by_tool is None
                else downsampled_cpgs_by_tool.get(t)
            )
            downsampled_mask_per_tool[t], downsampled_reason_per_tool[t] = (
                self._match_removed_cpgs_to_plot(
                    coord_index,
                    downsampled_df,
                    ch_col=ch_col,
                    x_col=x_col,
                    end_col=end_col,
                    default_reason="downsampled to HM450K universe",
                )
            )

        # consistent downsampling indices across panels
        n = len(df_chr)
        if n > max_points:
            rng = np.random.default_rng(42)
            keep_idx = np.sort(rng.choice(n, size=max_points, replace=False))
            downsampled = True
        else:
            keep_idx = np.arange(n)
            downsampled = False

        # figure size scaling: roughly 300 px per panel (adjust as desired), with min height
        height_per_panel = 300
        total_height = max(450, height_per_panel * n_tools)

        # Create stacked figure (n_tools rows x 1 col)
        fig = make_subplots(
            rows=n_tools,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f"{t}" for t in tools],
        )

        plot_state_order = ["Downsampled", "Removed", "not PMD", "PMD"]
        palette = {
            "Downsampled": "#9E9E9E",
            "Removed": "#FBEDA7",
            "not PMD": "#0400FF",
            "PMD": "#D32F2F",
        }
        shown_legend_names: set[str] = set()

        # Add traces for each tool in its own row
        for row, tool in enumerate(tools, start=1):
            df_plot = df_chr.iloc[keep_idx].copy().reset_index(drop=True)
            df_plot["state"] = states_per_tool[tool][keep_idx].astype(int)
            df_plot["is_downsampled"] = downsampled_mask_per_tool[tool][keep_idx]
            df_plot["is_removed"] = removed_mask_per_tool[tool][keep_idx]
            df_plot["removed_reason"] = np.where(
                df_plot["is_downsampled"],
                downsampled_reason_per_tool[tool][keep_idx],
                removed_reason_per_tool[tool][keep_idx],
            )
            df_plot["plot_state"] = np.where(
                df_plot["is_downsampled"],
                "Downsampled",
                np.where(
                    df_plot["is_removed"],
                    "Removed",
                    np.where(df_plot["state"] == 1, "PMD", "not PMD"),
                ),
            )

            for plot_state in plot_state_order:
                g = df_plot[df_plot["plot_state"] == plot_state]
                if g.empty:
                    continue

                hover_name = f"{tool} {plot_state}"
                showlegend = plot_state not in shown_legend_names
                shown_legend_names.add(plot_state)

                scatter_kwargs = {
                    "x": g[x_col],
                    "y": g[y_col],
                    "mode": "markers",
                    "marker": dict(size=point_size, color=palette[plot_state]),
                    "opacity": 0.55,
                    "name": plot_state,
                    "showlegend": showlegend,
                }

                if plot_state in {"Removed", "Downsampled"}:
                    scatter_kwargs["customdata"] = g[["removed_reason"]].to_numpy()
                    scatter_kwargs["hovertemplate"] = (
                        "pos: %{x}<br>"
                        "beta: %{y:.3f}<br>"
                        f"state: {plot_state}<br>"
                        "reason: %{customdata[0]}<extra>"
                        f"{hover_name}</extra>"
                    )
                else:
                    scatter_kwargs["hovertemplate"] = (
                        "pos: %{x}<br>"
                        "beta: %{y:.3f}<br>"
                        f"state: {plot_state}<extra>"
                        f"{hover_name}</extra>"
                    )

                fig.add_trace(
                    go.Scattergl(**scatter_kwargs),
                    row=row,
                    col=1,
                )

            # y-axis for this row
            fig.update_yaxes(
                title_text="Methylation (beta)",
                range=[-0.05, 1.05],
                row=row,
                col=1,
                showgrid=False,
            )

        # Shared x-axis with range slider on bottom subplot
        fig.update_xaxes(
            title_text="Genomic position",
            rangeslider=dict(visible=True),
            type="linear",
            row=n_tools,
            col=1,
            showgrid=False,
        )

        fig.update_layout(
            title=(
                f"{sample_name + ' - ' if sample_name else ''}"
                f"{chrom}: CpG methylation by tool ({'downsampled' if downsampled else 'full'})"
            ),
            height=total_height,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01
            ),
            margin=dict(l=60, r=20, t=80, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        if html_out is not None:
            fig.write_html(html_out)

        return fig, df_chr

    def _create_chromosome_plots(self, chrom):
        wgbs_chrom_plot_file = (
            Path(self.out_dir) / f"{chrom}_multi_tool_wgbs_pmd_labels.html"
        )
        methylseg = self.pathways["MethylSeg"]
        wgbs_removed_cpgs_df = methylseg.load_removed_cpgs("wgbs")
        hm450_removed_cpgs_df = methylseg.load_removed_cpgs("hm450k")

        if not wgbs_chrom_plot_file.exists() or self.force_recreate:
            wgbs_meth_df = self._load_plot_universe("wgbs")
            hm450k_meth_df = self._load_plot_universe("hm450k")
            wgbs_removed_cpgs = {
                "methylseg_wgbs": wgbs_removed_cpgs_df,
                "methylseg_hm450k": hm450_removed_cpgs_df,
            }
            downsampled_cpgs = self._build_reverse_intersection_removed_cpgs(
                wgbs_meth_df,
                hm450k_meth_df,
                reason="downsampled to HM450K universe",
            )
            wgbs_downsampled_cpgs = {
                "methylseg_hm450k": downsampled_cpgs,
                "dnmtools_array": downsampled_cpgs,
            }

            fig, df_chr = self._plot_chromosome_tools(
                meth_df=wgbs_meth_df,
                chrom=chrom,
                tools=list(self.region_beds.keys()),
                beds_dict=self.region_beds,
                sample_name=self.sample_id,
                max_points=220_000,
                html_out=wgbs_chrom_plot_file,
                removed_cpgs_by_tool=wgbs_removed_cpgs,
                downsampled_cpgs_by_tool=wgbs_downsampled_cpgs,
            )

        hm450k_chrom_plot_file = (
            Path(self.out_dir) / f"{chrom}_multi_tool_450k_pmd_labels.html"
        )

        if not hm450k_chrom_plot_file.exists() or self.force_recreate:
            hm450k_meth_df = self._load_plot_universe("hm450k")
            hm450k_removed_cpgs = {
                "methylseg_wgbs": wgbs_removed_cpgs_df,
                "methylseg_hm450k": hm450_removed_cpgs_df,
            }

            fig, df_chr = self._plot_chromosome_tools(
                meth_df=hm450k_meth_df,
                chrom=chrom,
                tools=list(self.region_beds.keys()),
                beds_dict=self.region_beds,
                sample_name=self.sample_id,
                max_points=120_000,
                html_out=hm450k_chrom_plot_file,
                removed_cpgs_by_tool=hm450k_removed_cpgs,
            )

    def _create_run_stat_plots(self):
        memory_plot_path = Path(self.out_dir) / "memory_barplot.html"
        runtime_plot_path = Path(self.out_dir) / "runtime_barplot.html"

        if (
            memory_plot_path.exists() and runtime_plot_path.exists()
        ) and not self.force_recreate:
            return {
                "runtime_plot": str(runtime_plot_path),
                "memory_plot": str(memory_plot_path),
            }

        records = []

        for tool_name, pathway in self.pathways.items():
            run_stats_dir = Path(pathway.out_dir) / self.sample_id / "run_stats"
            print(f"Looking for run stats in {run_stats_dir}")

            if not run_stats_dir.exists():
                continue

            for stats_file in run_stats_dir.glob("*_run_stats.csv"):
                print("reading file", stats_file)
                df = pd.read_csv(stats_file)

                if df.empty:
                    continue

                if df.iloc[0].get("cache_used", False) is True:
                    print("cache used — unable to make plot")
                    return False

                for _, row in df.iterrows():
                    records.append(
                        {
                            "tool": tool_name,  # parent tool
                            "subtask": row["tool_name"],  # subtask name
                            "runtime_sec": row["runtime_sec"],
                            "mem_peak_mb": row["mem_peak_mb"],
                        }
                    )

        if not records:
            print("No run stats found — skipping plots")
            return False

        run_stats_df = pd.DataFrame(records)

        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # Runtime bar plot (one bar per subtask)
        # -------------------------------------------------
        fig_runtime = px.bar(
            run_stats_df,
            x="subtask",
            y="runtime_sec",
            color="tool",  # color by parent tool (optional but helpful)
            title="Runtime by subtask",
            labels={
                "runtime_sec": "Runtime (seconds)",
                "subtask": "Subtask",
                "tool": "Tool",
            },
            text_auto=".2s",
        )

        fig_runtime.update_layout(
            template="plotly_white",
            xaxis_tickangle=30,
            barmode="group",  # <-- key: grouped, not stacked
        )

        fig_runtime.write_html(
            runtime_plot_path,
            include_plotlyjs="cdn",
        )

        # -------------------------------------------------
        # Peak memory bar plot (one bar per subtask)
        # -------------------------------------------------
        fig_mem = px.bar(
            run_stats_df,
            x="subtask",
            y="mem_peak_mb",
            color="tool",
            title="Peak memory usage by subtask",
            labels={
                "mem_peak_mb": "Peak memory (MB)",
                "subtask": "Subtask",
                "tool": "Tool",
            },
            text_auto=".2s",
        )

        fig_mem.update_layout(
            template="plotly_white",
            xaxis_tickangle=30,
            barmode="group",  # <-- key: grouped bars
        )

        fig_mem.write_html(
            memory_plot_path,
            include_plotlyjs="cdn",
        )

        print(f"Wrote runtime and memory bar plots to {out_dir}")
        return {
            "runtime_plot": str(runtime_plot_path),
            "memory_plot": str(memory_plot_path),
        }

    def _generate_stats_for_aggregate_summaries(self):
        aggregate_summaries = Path(self.out_dir) / "aggregate_summaries"
        aggregate_summaries.mkdir(parents=True, exist_ok=True)
        run_stats_out = aggregate_summaries / "all_run_stats.csv"
        region_stats_out = aggregate_summaries / "all_region_stats.csv"
        region_context_out = aggregate_summaries / "all_region_context_stats.csv"

        if (
            run_stats_out.exists()
            and region_stats_out.exists()
            and region_context_out.exists()
            and not self.force_recreate
        ):
            return {
                "all_run_stats": str(run_stats_out),
                "all_region_stats": str(region_stats_out),
                "all_region_context_stats": str(region_context_out),
            }

        if not hasattr(self, "region_dfs"):
            self._load_regions()

        def _build_methylseg_platform_data(platform: str) -> dict:
            methylseg_out_dir = (
                Path(self.pathways["MethylSeg"].out_dir)
                / self.sample_id
                / "out"
                / platform
            )
            meth_ref_name = (
                "450k_meth_ref.tsv" if platform == "hm450k" else "wgbs_meth_ref.tsv"
            )
            meth_ref_path = (
                Path(self.pathways["MethylSeg"].out_dir)
                / self.sample_id
                / "prep"
                / meth_ref_name
            )

            meth_by_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if meth_ref_path.exists():
                meth_df = pd.read_csv(meth_ref_path, sep="\t")
                meth_df.columns = ["CpG_chrm", "CpG_beg", "CpG_end", "beta"]
                meth_df = meth_df.sort_values(["CpG_chrm", "CpG_beg"])
                for chrom, chrom_df in meth_df.groupby("CpG_chrm", sort=False):
                    meth_by_chrom[chrom] = (
                        chrom_df["CpG_beg"].to_numpy(dtype=np.int64),
                        chrom_df["beta"].to_numpy(dtype=float),
                    )

            state_intervals: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            for state in MethylationStates._member_names_:
                state_file = (
                    methylseg_out_dir
                    / "summary_files"
                    / f"segments_cleaned_{state}.bed"
                )
                if state_file.exists() and state_file.stat().st_size > 0:
                    state_df = pd.read_csv(
                        state_file,
                        sep="\t",
                        header=None,
                        names=["chr", "start", "end", "label"],
                    )
                else:
                    state_df = pd.DataFrame(columns=["chr", "start", "end", "label"])
                if not state_df.empty:
                    state_df = state_df.sort_values(
                        ["chr", "start", "end"]
                    ).reset_index(drop=True)

                chrom_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                if not state_df.empty:
                    for chrom, chrom_df in state_df.groupby("chr", sort=False):
                        chrom_intervals[chrom] = (
                            chrom_df["start"].to_numpy(dtype=np.int64),
                            chrom_df["end"].to_numpy(dtype=np.int64),
                        )
                state_intervals[state] = chrom_intervals

            return {
                "meth_by_chrom": meth_by_chrom,
                "state_intervals": state_intervals,
            }

        def _region_type_for_tool(tool_name: str) -> str:
            if "pmr" in tool_name.lower():
                return "PMR"
            return "PMD"

        def _platform_for_tool(tool_name: str) -> str:
            if tool_name in {"dnmtools_array", "methylseg_hm450k"}:
                return "hm450k"
            return "wgbs"

        def _get_overlap_bp(
            start: int,
            end: int,
            interval_starts: np.ndarray,
            interval_ends: np.ndarray,
        ) -> int:
            if interval_starts.size == 0 or end <= start:
                return 0
            left = np.searchsorted(interval_ends, start, side="right")
            right = np.searchsorted(interval_starts, end, side="left")
            if right <= left:
                return 0
            overlap_starts = np.maximum(interval_starts[left:right], start)
            overlap_ends = np.minimum(interval_ends[left:right], end)
            overlaps = np.clip(overlap_ends - overlap_starts, a_min=0, a_max=None)
            return int(overlaps.sum())

        platform_data = {
            "wgbs": _build_methylseg_platform_data("wgbs"),
            "hm450k": _build_methylseg_platform_data("hm450k"),
        }
        methylseg_pathway = self.pathways["MethylSeg"]
        int_low_cutoff = float(methylseg_pathway.int_low_cutoff)
        int_high_cutoff = float(methylseg_pathway.int_high_cutoff)

        run_stat_frames = []
        for tool_name, pathway in self.pathways.items():
            run_stats_dir = Path(pathway.out_dir) / self.sample_id / "run_stats"
            if not run_stats_dir.exists():
                continue
            for stats_file in sorted(run_stats_dir.glob("*_run_stats.csv")):
                stats_df = pd.read_csv(stats_file)
                if stats_df.empty:
                    continue
                stats_df = stats_df.copy()
                stats_df["sample_id"] = self.sample_id
                stats_df["tool"] = tool_name
                stats_df["subtask"] = stats_df["tool_name"]
                stats_df["source_file"] = str(stats_file)
                run_stat_frames.append(stats_df)

        if run_stat_frames:
            pd.concat(run_stat_frames, ignore_index=True).to_csv(
                run_stats_out, index=False
            )
        else:
            pd.DataFrame(
                columns=[
                    "tool_name",
                    "runtime_sec",
                    "mem_peak_mb",
                    "cache_used",
                    "sample_id",
                    "tool",
                    "subtask",
                    "source_file",
                ]
            ).to_csv(run_stats_out, index=False)

        region_stats_file = Path(self.out_dir) / "region_stats.tsv"
        if region_stats_file.exists():
            region_stats_df = pd.read_csv(region_stats_file, sep="\t", index_col=0)
        else:
            region_stats_df = self._get_stats()
            region_stats_df.to_csv(region_stats_file, sep="\t")

        region_stats_df = region_stats_df.reset_index().rename(
            columns={"index": "tool"}
        )
        region_stats_df["sample_id"] = self.sample_id
        region_stats_df["source_file"] = str(region_stats_file)
        region_stats_df.to_csv(region_stats_out, index=False)

        region_context_frames = {
            name: df.copy() for name, df in self.region_dfs.items()
        }

        for platform in ["wgbs", "hm450k"]:
            pmd_intervals = platform_data[platform]["state_intervals"]["PMD"]
            pmd_rows = []
            for chrom, (starts, ends) in pmd_intervals.items():
                if starts.size == 0:
                    continue
                pmd_rows.append(
                    pd.DataFrame(
                        {
                            "chr": chrom,
                            "start": starts,
                            "end": ends,
                        }
                    )
                )
            region_context_frames[f"methylseg_{platform}"] = (
                pd.concat(pmd_rows, ignore_index=True)
                if pmd_rows
                else pd.DataFrame(columns=["chr", "start", "end"])
            )

        context_columns = [
            "sample_id",
            "tool",
            "platform",
            "region_type",
            "chrom",
            "start",
            "end",
            "region_length_bp",
            "n_cpg",
            "mean_meth",
            "median_meth",
            "meth_std",
            "n_low_cpg",
            "n_intermediate_cpg",
            "n_high_cpg",
            "pct_low",
            "pct_intermediate",
            "pct_high",
        ]
        context_records = []
        for tool_name, region_df in region_context_frames.items():
            if region_df.empty:
                continue

            platform = _platform_for_tool(tool_name)
            meth_by_chrom = platform_data[platform]["meth_by_chrom"]
            tool_df = region_df.loc[:, ["chr", "start", "end"]].copy()
            tool_df["start"] = tool_df["start"].astype(np.int64)
            tool_df["end"] = tool_df["end"].astype(np.int64)
            tool_df = tool_df.sort_values(["chr", "start", "end"]).reset_index(
                drop=True
            )

            for chrom, chrom_df in tool_df.groupby("chr", sort=False):
                starts = chrom_df["start"].to_numpy(dtype=np.int64)
                ends = chrom_df["end"].to_numpy(dtype=np.int64)
                lengths = np.clip(ends - starts, a_min=0, a_max=None)

                meth_positions, meth_values = meth_by_chrom.get(
                    chrom,
                    (np.array([], dtype=np.int64), np.array([], dtype=float)),
                )
                left_idx = np.searchsorted(meth_positions, starts, side="left")
                right_idx = np.searchsorted(meth_positions, ends, side="left")

                for idx, (row, region_start, region_end, region_len) in enumerate(
                    zip(chrom_df.itertuples(index=False), starts, ends, lengths)
                ):
                    meth_slice = meth_values[left_idx[idx] : right_idx[idx]]
                    n_cpg = int(meth_slice.size)
                    meth_std = (
                        float(np.std(meth_slice, ddof=1))
                        if meth_slice.size > 1
                        else np.nan
                    )

                    low_mask = meth_slice < int_low_cutoff
                    intermediate_mask = (meth_slice >= int_low_cutoff) & (
                        meth_slice <= int_high_cutoff
                    )
                    high_mask = meth_slice > int_high_cutoff

                    n_low_cpg = int(low_mask.sum())
                    n_intermediate_cpg = int(intermediate_mask.sum())
                    n_high_cpg = int(high_mask.sum())
                    context_records.append(
                        {
                            "sample_id": self.sample_id,
                            "tool": tool_name,
                            "platform": platform,
                            "region_type": _region_type_for_tool(tool_name),
                            "chrom": row.chr,
                            "start": int(region_start),
                            "end": int(region_end),
                            "region_length_bp": int(region_len),
                            "n_cpg": int(meth_slice.size),
                            "mean_meth": (
                                float(meth_slice.mean()) if meth_slice.size else np.nan
                            ),
                            "median_meth": (
                                float(np.median(meth_slice))
                                if meth_slice.size
                                else np.nan
                            ),
                            "meth_std": meth_std,
                            "n_low_cpg": n_low_cpg,
                            "n_intermediate_cpg": n_intermediate_cpg,
                            "n_high_cpg": n_high_cpg,
                            "pct_low": (n_low_cpg * 100.0 / n_cpg if n_cpg else np.nan),
                            "pct_intermediate": (
                                n_intermediate_cpg * 100.0 / n_cpg if n_cpg else np.nan
                            ),
                            "pct_high": (
                                n_high_cpg * 100.0 / n_cpg if n_cpg else np.nan
                            ),
                        }
                    )

        pd.DataFrame(context_records, columns=context_columns).to_csv(
            region_context_out, index=False
        )
        return {
            "all_run_stats": str(run_stats_out),
            "all_region_stats": str(region_stats_out),
            "all_region_context_stats": str(region_context_out),
        }

    def run_comparison(self):
        self._load_regions()

        stats_file = Path(self.out_dir) / f"region_stats.tsv"
        if not stats_file.exists() or self.force_recreate:
            stats = self._get_stats()

            stats.to_csv(
                Path(self.out_dir) / f"region_stats.tsv",
                sep="\t",
            )

        self._create_heatmaps()

        for chrom in [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]:
            self._create_chromosome_plots(chrom=chrom)

        self._create_run_stat_plots()
        self._generate_stats_for_aggregate_summaries()

    def run(self):
        results = self.run_all()
        self.run_comparison()
        return results


class MethylToolAggregateComparator:
    def __init__(self, comparison_out_dirs: Sequence[str | Path]):
        self.comparison_out_dirs = [Path(out_dir) for out_dir in comparison_out_dirs]

    def generate_aggregate_summary_plots(self):
        if not self.comparison_out_dirs:
            return {}

        first_out_dir = self.comparison_out_dirs[0]
        aggregate_out_dir = (
            first_out_dir.parent.parent / "aggregate_summaries"
            if first_out_dir.name == "aggregate_summaries"
            else first_out_dir.parent / "aggregate_summaries"
        )
        aggregate_out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        legacy_outputs = [
            "region_summary_table.html",
            "region_metric_heatmap.html",
            "region_metric_scatter.html",
            "region_context_summary_table.html",
            "region_context_metric_heatmap.html",
            "region_context_dominant_state_distribution.html",
            "run_summary_table.html",
            "run_runtime_heatmap.html",
            "run_memory_heatmap.html",
            "tool_similarity_pca.html",
            "tool_similarity_heatmap.html",
        ]
        for legacy_name in legacy_outputs:
            legacy_path = aggregate_out_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        def _read_csv(csv_path: Path) -> pd.DataFrame:
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                return pd.DataFrame()
            try:
                return pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        def _write_html(fig: go.Figure, out_name: str) -> Path:
            out_path = aggregate_out_dir / out_name
            fig.write_html(out_path, include_plotlyjs="cdn")
            outputs[out_name] = str(out_path)
            return out_path

        def _coerce_bool(series: pd.Series) -> pd.Series:
            return series.map(
                lambda value: (
                    value
                    if isinstance(value, bool)
                    else str(value).strip().lower() in {"true", "1", "yes"}
                )
            )

        def _write_summary_csv(
            df: pd.DataFrame,
            *,
            group_col: str,
            metrics: Sequence[str],
            out_name: str,
            excluded_counts: Optional[dict[str, int]] = None,
        ) -> pd.DataFrame:
            summary_rows = []
            for metric in metrics:
                if metric not in df.columns:
                    continue
                metric_df = df.loc[:, [group_col, metric]].copy()
                metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
                metric_df = metric_df.dropna(subset=[group_col, metric])
                if metric_df.empty:
                    continue

                for group_name, group_df in metric_df.groupby(group_col, sort=False):
                    summary_rows.append(
                        {
                            "metric": metric,
                            "group_type": group_col,
                            "group": str(group_name),
                            "n_samples": int(group_df.shape[0]),
                            "mean": float(group_df[metric].mean()),
                            "median": float(group_df[metric].median()),
                            "max": float(group_df[metric].max()),
                            "n_excluded_cache_rows": (
                                int(excluded_counts.get(str(group_name), 0))
                                if excluded_counts is not None
                                else np.nan
                            ),
                        }
                    )

            summary_cols = [
                "metric",
                "group_type",
                "group",
                "n_samples",
                "mean",
                "median",
                "max",
                "n_excluded_cache_rows",
            ]
            summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
            if excluded_counts is None:
                summary_df = summary_df.drop(columns=["n_excluded_cache_rows"])

            out_path = aggregate_out_dir / out_name
            summary_df.to_csv(out_path, index=False)
            outputs[out_name] = str(out_path)
            return summary_df

        def _write_box_plot(
            df: pd.DataFrame,
            *,
            group_col: str,
            metric: str,
            out_name: str,
            title: str,
        ) -> None:
            if metric not in df.columns or group_col not in df.columns:
                return

            plot_df = df.copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna(subset=[group_col, metric])
            if plot_df.empty:
                return
            plot_df[group_col] = plot_df[group_col].astype(str).str.strip()

            group_order = [str(group) for group in pd.unique(plot_df[group_col])]
            palette = px.colors.qualitative.Plotly
            fig = go.Figure()

            for idx, group_name in enumerate(group_order):
                group_df = plot_df.loc[plot_df[group_col] == group_name].copy()
                if group_df.empty:
                    continue

                hovertemplate = (
                    f"{group_col}: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>"
                )
                customdata = None
                if "sample_id" in group_df.columns:
                    customdata = group_df[["sample_id"]].to_numpy()
                    hovertemplate = (
                        f"{group_col}: %{{x}}<br>"
                        "sample_id: %{customdata[0]}<br>"
                        f"{metric}: %{{y:.4f}}<extra></extra>"
                    )

                color = palette[idx % len(palette)]
                fig.add_trace(
                    go.Box(
                        x=[group_name] * len(group_df),
                        y=group_df[metric],
                        name=group_name,
                        legendgroup=group_name,
                        boxpoints="all",
                        jitter=0.15,
                        pointpos=0.0,
                        marker=dict(size=5, opacity=0.55, color=color),
                        line=dict(color=color),
                        fillcolor=color,
                        opacity=0.65,
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                        showlegend=False,
                    )
                )

            max_df = plot_df.groupby(group_col, sort=False)[metric].max().reset_index()
            max_df[group_col] = max_df[group_col].astype(str).str.strip()
            if not max_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=max_df[group_col],
                        y=max_df[metric],
                        mode="markers",
                        name="max",
                        marker=dict(symbol="diamond", size=11, color="black"),
                        hovertemplate=(
                            f"{group_col}: %{{x}}<br>"
                            f"max {metric}: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

            data_min = plot_df[metric].min()
            data_max = plot_df[metric].max()
            y_min = max(0, data_min) if "runtime" in metric.lower() else data_min
            y_max = data_max * 1.05 if data_max != 0 else data_max + 1

            fig.update_layout(
                template="plotly_white",
                title=title,
                xaxis_title=group_col.replace("_", " ").title(),
                yaxis_title=metric,
                xaxis_tickangle=35,
                boxmode="group",
            )
            fig.update_xaxes(categoryorder="array", categoryarray=group_order)
            fig.update_yaxes(range=[y_min, y_max])
            _write_html(fig, out_name)

        region_frames = []
        context_frames = []
        run_frames = []
        missing_stats = []

        for comparison_out_dir in self.comparison_out_dirs:
            comparison_out_dir = Path(comparison_out_dir)
            sample_aggregate_dir = (
                comparison_out_dir
                if comparison_out_dir.name == "aggregate_summaries"
                else comparison_out_dir / "aggregate_summaries"
            )
            sample_id = (
                comparison_out_dir.parent.name
                if comparison_out_dir.name == "aggregate_summaries"
                else comparison_out_dir.name
            )
            stats_paths = {
                "all_region_stats": sample_aggregate_dir / "all_region_stats.csv",
                "all_region_context_stats": sample_aggregate_dir
                / "all_region_context_stats.csv",
                "all_run_stats": sample_aggregate_dir / "all_run_stats.csv",
            }

            missing_for_sample = [
                str(path)
                for path in stats_paths.values()
                if not path.exists() or path.stat().st_size == 0
            ]
            if missing_for_sample:
                missing_stats.extend(missing_for_sample)
                continue

            region_df = _read_csv(stats_paths["all_region_stats"])
            if not region_df.empty:
                if "sample_id" not in region_df.columns:
                    region_df["sample_id"] = sample_id
                region_frames.append(region_df)

            context_df = _read_csv(stats_paths["all_region_context_stats"])
            if not context_df.empty:
                if "sample_id" not in context_df.columns:
                    context_df["sample_id"] = sample_id
                context_frames.append(context_df)

            run_df = _read_csv(stats_paths["all_run_stats"])
            if not run_df.empty:
                if "sample_id" not in run_df.columns:
                    run_df["sample_id"] = sample_id
                run_frames.append(run_df)

        if missing_stats:
            missing_stats_str = "\n".join(sorted(set(missing_stats)))
            raise FileNotFoundError(
                "Missing aggregate summary CSVs. Run "
                "_generate_stats_for_aggregate_summaries for each comparison out_dir first:\n"
                f"{missing_stats_str}"
            )

        region_stats = (
            pd.concat(region_frames, ignore_index=True).drop_duplicates()
            if region_frames
            else pd.DataFrame()
        )
        context_stats = (
            pd.concat(context_frames, ignore_index=True).drop_duplicates()
            if context_frames
            else pd.DataFrame()
        )
        run_stats = (
            pd.concat(run_frames, ignore_index=True).drop_duplicates()
            if run_frames
            else pd.DataFrame()
        )

        region_csv = aggregate_out_dir / "all_region_stats.csv"
        context_csv = aggregate_out_dir / "all_region_context_stats.csv"
        run_csv = aggregate_out_dir / "all_run_stats.csv"

        region_stats.to_csv(region_csv, index=False)
        context_stats.to_csv(context_csv, index=False)
        run_stats.to_csv(run_csv, index=False)

        outputs["all_region_stats.csv"] = str(region_csv)
        outputs["all_region_context_stats.csv"] = str(context_csv)
        outputs["all_run_stats.csv"] = str(run_csv)

        run_metrics = ["runtime_sec", "mem_peak_mb"]
        region_metrics = [
            "n_regions",
            "total_bp",
            "genome_coverage_pct",
            "mean_length",
            "median_length",
            "std_length",
            "min_length",
            "max_length",
        ]
        context_metrics = [
            "region_length_bp",
            "n_cpg",
            "mean_meth",
            "median_meth",
            "meth_std",
            "pct_low",
            "pct_intermediate",
            "pct_high",
        ]
        run_stats_for_summary = run_stats.copy()
        if not run_stats_for_summary.empty:
            for metric in run_metrics:
                if metric in run_stats_for_summary.columns:
                    run_stats_for_summary[metric] = pd.to_numeric(
                        run_stats_for_summary[metric], errors="coerce"
                    )
            if "cache_used" in run_stats_for_summary.columns:
                run_stats_for_summary["cache_used"] = _coerce_bool(
                    run_stats_for_summary["cache_used"]
                )
            else:
                run_stats_for_summary["cache_used"] = False

        excluded_cache_counts = (
            run_stats_for_summary.loc[run_stats_for_summary["cache_used"]]
            .groupby("subtask")
            .size()
            .astype(int)
            .to_dict()
            if not run_stats_for_summary.empty
            else {}
        )
        usable_run_stats = (
            run_stats_for_summary.loc[~run_stats_for_summary["cache_used"]].copy()
            if not run_stats_for_summary.empty
            else pd.DataFrame(columns=["sample_id", "subtask"] + run_metrics)
        )

        region_stats_for_summary = region_stats.copy()
        for metric in region_metrics:
            if metric in region_stats_for_summary.columns:
                region_stats_for_summary[metric] = pd.to_numeric(
                    region_stats_for_summary[metric], errors="coerce"
                )

        available_context_metrics = [
            metric for metric in context_metrics if metric in context_stats.columns
        ]
        if not context_stats.empty and available_context_metrics:
            context_stats_for_summary = (
                context_stats.groupby(["sample_id", "tool"], sort=False)[
                    available_context_metrics
                ]
                .mean()
                .reset_index()
            )
        else:
            context_stats_for_summary = pd.DataFrame(
                columns=["sample_id", "tool"] + available_context_metrics
            )

        _write_summary_csv(
            usable_run_stats,
            group_col="subtask",
            metrics=run_metrics,
            out_name="run_stat_summary.csv",
            excluded_counts=excluded_cache_counts,
        )
        _write_summary_csv(
            region_stats_for_summary,
            group_col="tool",
            metrics=region_metrics,
            out_name="region_stat_summary.csv",
        )
        _write_summary_csv(
            context_stats_for_summary,
            group_col="tool",
            metrics=available_context_metrics,
            out_name="region_context_stat_summary.csv",
        )

        for metric in run_metrics:
            _write_box_plot(
                usable_run_stats,
                group_col="subtask",
                metric=metric,
                out_name=f"run_{metric}_box.html",
                title=f"Cross-sample distribution of {metric} by run subtask",
            )

        for metric in region_metrics:
            _write_box_plot(
                region_stats_for_summary,
                group_col="tool",
                metric=metric,
                out_name=f"region_{metric}_box.html",
                title=f"Cross-sample distribution of {metric} by tool",
            )

        for metric in available_context_metrics:
            _write_box_plot(
                context_stats_for_summary,
                group_col="tool",
                metric=metric,
                out_name=f"context_{metric}_box.html",
                title=(
                    f"Cross-sample distribution of per-sample mean {metric} by tool"
                ),
            )

        return outputs
