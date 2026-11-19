# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.2",
#     "gdrive-fsspec==0.4.1",
#     "google-api-python-client>=2,<3",
#     "huggingface-hub>=2,<3",
#     "httpx>=0.28,<1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import base64
    import hashlib
    import io
    import json
    import os
    import shutil
    import stat
    import subprocess
    import tarfile
    import uuid
    import zipfile
    from contextlib import closing
    from pathlib import Path, PurePosixPath
    from tempfile import TemporaryDirectory, gettempdir

    import httpx
    import marimo as mo
    from gdrive_fsspec import GoogleDriveFileSystem
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
    from huggingface_hub import HfApi, get_token, hf_hub_url


@app.cell
def _():
    REPO_URL = "https://github.com/chillCode404/OASIS-testing.git"
    WORKSPACE = Path(gettempdir()) / "oasis-molab"
    REPO_DIR = WORKSPACE / "OASIS-testing"
    SCRATCH_DIR = WORKSPACE / "staging"
    MIB = 1024**2
    BLOCK_SIZE = 4 * MIB
    DISK_RESERVE = 256 * MIB
    BENCHMARKS = {
        "OVO-Bench": {
            "repo_id": "JoeLeelyf/OVO-Bench",
            "revision": "fec29e3385747b5642d995370143ba92d2819bd2",
            "metadata": ("eval_ovo_bench.json",),
        },
        "StreamBench": {
            "repo_id": "Barry-12138/StreamBench_v0.3",
            "revision": "739fbcbff02b3aaebaef1795c4be8b717777c749",
            "metadata": ("eval_streambench.json",),
        },
        "StreamingBench": {
            "repo_id": "mjuicem/StreamingBench",
            "revision": "48872fa707124474ce4c5172ddc58efb8bc88058",
            "metadata": (
                "StreamingBench/Anomaly_Context_Understanding.json",
                "StreamingBench/Misleading_Context_Recognition.json",
                "StreamingBench/Real_Time_Visual_Understanding.json",
                "StreamingBench/Sequential_Question_Answering.json",
            ),
        },
    }
    return BENCHMARKS, BLOCK_SIZE, DISK_RESERVE, MIB, REPO_DIR, REPO_URL, SCRATCH_DIR, WORKSPACE


@app.cell
def _(REPO_DIR, REPO_URL, WORKSPACE):
    def clone_oasis():
        def git(*args, env=None):
            return subprocess.run(
                ["git", *args], env=env, capture_output=True, text=True, check=True
            ).stdout.strip()

        WORKSPACE.mkdir(parents=True, exist_ok=True)
        if not REPO_DIR.exists():
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            token = env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")
            if token:
                # Scope credentials to this repository and keep them out of the URL.
                count = int(env.get("GIT_CONFIG_COUNT", "0"))
                credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
                env.update(
                    {
                        "GIT_CONFIG_COUNT": str(count + 1),
                        f"GIT_CONFIG_KEY_{count}": f"http.{REPO_URL}.extraHeader",
                        f"GIT_CONFIG_VALUE_{count}": f"Authorization: Basic {credential}",
                    }
                )
            with TemporaryDirectory(dir=WORKSPACE, prefix="clone-") as temporary:
                checkout = Path(temporary) / REPO_DIR.name
                try:
                    git("clone", "--depth", "1", "--", REPO_URL, str(checkout), env=env)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        "Git clone failed. Check repository access and add GITHUB_TOKEN "
                        "to molab Secrets if the repository requires authentication."
                    ) from error
                checkout.rename(REPO_DIR)

        if not (REPO_DIR / ".git").exists():
            raise ValueError(f"{REPO_DIR} is already occupied by a non-Git directory.")
        origin = git("-C", str(REPO_DIR), "config", "--get", "remote.origin.url")
        if origin.rstrip("/").removesuffix(".git") != REPO_URL.removesuffix(".git"):
            raise ValueError(f"The checkout at {REPO_DIR} has an unexpected origin.")
        return git("-C", str(REPO_DIR), "rev-parse", "HEAD")

    return (clone_oasis,)


@app.cell
def _(BENCHMARKS, REPO_DIR):
    def required_videos(benchmark):
        paths = set()
        for metadata in BENCHMARKS[benchmark]["metadata"]:
            with (REPO_DIR / "metadata" / metadata).open(encoding="utf-8") as source:
                rows = json.load(source)
            for row in rows:
                path = PurePosixPath(row["info"]["video_path"])
                if (
                    len(path.parts) < 3
                    or path.parts[0] != benchmark
                    or ".." in path.parts
                    or path.suffix.lower() != ".mp4"
                ):
                    raise ValueError(f"Unexpected path in {metadata}: {path}")
                paths.add(path)
        if not paths:
            raise ValueError(f"No videos found in the {benchmark} metadata.")
        return paths

    def zip_name(path):
        benchmark, group = path.parts[:2]
        if benchmark == "StreamBench":
            return f"{group}.zip"

        label = group.replace("_", " ")
        group_size = {
            "Real-Time_Visual_Understanding": 50,
            "Sequential_Question_Answering": 25,
        }.get(group)
        if group_size is None:
            return f"{label}.zip"
        sample = int(path.parent.name.removeprefix("sample_"))
        first = (sample - 1) // group_size * group_size + 1
        return f"{label}_{first}-{first + group_size - 1}.zip"

    def archive_sizes(benchmark, paths):
        spec = BENCHMARKS[benchmark]
        if benchmark == "OVO-Bench":
            names = [f"chunked_videos.tar.parta{part}" for part in "abcdefghijklmno"]
        else:
            names = sorted({zip_name(path) for path in paths})
        api = HfApi(token=get_token() if benchmark == "StreamBench" else False)
        items = api.get_paths_info(
            spec["repo_id"], names, repo_type="dataset", revision=spec["revision"]
        )
        sizes = {entry.path: entry.size for entry in items if getattr(entry, "size", 0)}
        missing = set(names) - sizes.keys()
        if missing:
            raise RuntimeError(f"Missing archives at the pinned revision: {', '.join(sorted(missing))}")
        return {name: sizes[name] for name in names}

    return archive_sizes, required_videos, zip_name


@app.cell
def _(BLOCK_SIZE):
    class DriveDataset:
        folder_type = "application/vnd.google-apps.folder"
        fields = "id,name,mimeType,size,md5Checksum,appProperties"

        def __init__(self, filesystem, destination):
            self.files = filesystem.files
            self.service = filesystem.srv
            self.listings = {}
            self.root_id = self.folder(filesystem.root_file_id, destination, create=True)

        def children(self, parent_id):
            if parent_id in self.listings:
                return self.listings[parent_id]
            entries = {}
            page_token = None
            while True:
                page = self.files.list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    fields=f"nextPageToken,files({self.fields})",
                    pageSize=1000,
                    pageToken=page_token,
                ).execute(num_retries=4)
                for entry in page.get("files", []):
                    entries.setdefault(entry["name"], []).append(entry)
                page_token = page.get("nextPageToken")
                if not page_token:
                    self.listings[parent_id] = entries
                    return entries

        def lookup(self, parent_id, name):
            matches = self.children(parent_id).get(name, [])
            if len(matches) > 1:
                raise ValueError(f"Drive has duplicate entries named {name!r} in folder {parent_id}.")
            return matches[0] if matches else None

        def folder(self, parent_id, path, *, create=False):
            for part in path.parts:
                entry = self.lookup(parent_id, part)
                if entry is None:
                    if not create:
                        return None
                    entry = self.files.create(
                        body={"name": part, "mimeType": self.folder_type, "parents": [parent_id]},
                        fields=self.fields,
                    ).execute()
                    self.children(parent_id)[part] = [entry]
                if entry["mimeType"] != self.folder_type:
                    raise ValueError(f"Expected a Drive folder at {path}. Found a file named {part}.")
                parent_id = entry["id"]
            return parent_id

        def check_access(self):
            payload = uuid.uuid4().hex.encode()
            name = f".oasis-connection-{payload.decode()}.txt"
            response = self.files.create(
                body={"name": name, "parents": [self.root_id]},
                media_body=MediaIoBaseUpload(io.BytesIO(payload), mimetype="text/plain"),
                fields="id",
            ).execute()
            try:
                if self.files.get_media(fileId=response["id"]).execute(num_retries=4) != payload:
                    raise RuntimeError("Drive connection check failed its read-back.")
            finally:
                self.files.delete(fileId=response["id"]).execute(num_retries=4)

        def free_bytes(self):
            quota = self.service.about().get(fields="storageQuota").execute(num_retries=4)["storageQuota"]
            return int(quota["limit"]) - int(quota.get("usage", 0)) if "limit" in quota else None

        def inventory(self, paths):
            self.listings.clear()
            found = {}
            for path in sorted(paths):
                parent_id = self.folder(self.root_id, path.parent)
                if parent_id is not None:
                    entry = self.lookup(parent_id, path.name)
                    if entry is not None:
                        found[path] = entry
            return found

        def record_video(self, path, temporary_file, size, digest, source_key):
            parent_id = self.folder(self.root_id, path.parent, create=True)
            existing = self.lookup(parent_id, path.name)
            properties = {"oasisSource": source_key, "oasisMd5": digest}
            if existing is None:
                media = MediaFileUpload(
                    str(temporary_file), mimetype="video/mp4", chunksize=BLOCK_SIZE, resumable=True
                )
                try:
                    request = self.files.create(
                        body={"name": path.name, "parents": [parent_id], "appProperties": properties},
                        media_body=media,
                        fields=self.fields,
                    )
                    entry = None
                    while entry is None:
                        _, entry = request.next_chunk(num_retries=4)
                finally:
                    media.stream().close()
            else:
                if int(existing.get("size", 0)) != size or existing.get("md5Checksum") != digest:
                    raise ValueError(
                        f"Existing Drive file differs from the archive: {path}. "
                        "Move it aside or select another destination."
                    )
                entry = self.files.update(
                    fileId=existing["id"],
                    body={"appProperties": {**existing.get("appProperties", {}), **properties}},
                    fields=self.fields,
                ).execute(num_retries=4)

            if int(entry.get("size", 0)) != size or entry.get("md5Checksum") != digest:
                raise IOError(f"Drive did not verify the uploaded bytes for {path}.")
            self.children(parent_id)[path.name] = [entry]
            return entry

        def save_manifest(self, benchmark, manifest):
            parent_id = self.folder(self.root_id, PurePosixPath(benchmark), create=True)
            name = ".oasis-manifest.json"
            content = json.dumps(manifest, indent=2, sort_keys=True).encode()
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json")
            existing = self.lookup(parent_id, name)
            if existing is None:
                request = self.files.create(body={"name": name, "parents": [parent_id]}, media_body=media)
            else:
                request = self.files.update(fileId=existing["id"], media_body=media)
            request.execute()

    def verified_videos(inventory, source_key):
        verified = {}
        for path, entry in inventory.items():
            props = entry.get("appProperties", {})
            digest = entry.get("md5Checksum")
            if (
                props.get("oasisSource") == source_key
                and digest
                and props.get("oasisMd5") == digest
                and int(entry.get("size", 0)) > 0
            ):
                verified[path] = entry
        return verified

    return DriveDataset, verified_videos


@app.cell
def _(BLOCK_SIZE, DISK_RESERVE, MIB, SCRATCH_DIR):
    def format_bytes(size):
        return f"{size / 1e9:,.2f} GB" if size >= 1e9 else f"{size / 1e6:,.1f} MB"

    def check_scratch(size):
        free = shutil.disk_usage(SCRATCH_DIR).free
        if size + DISK_RESERVE > free:
            raise OSError(
                f"Scratch needs {format_bytes(size + DISK_RESERVE)} with the reserve. "
                f"Available: {format_bytes(free)}."
            )

    def archive_bytes(client, spec, name, size, token):
        url = hf_hub_url(spec["repo_id"], name, repo_type="dataset", revision=spec["revision"])
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        print(f"Reading {name} ({format_bytes(size)})", flush=True)
        received = 0
        next_report = 512 * MIB
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code in (401, 403):
                    raise PermissionError(
                        f"Hugging Face denied access to {name}. Check the dataset's access "
                        "conditions and the HF_TOKEN in molab Secrets."
                    )
                if response.status_code != 200:
                    raise OSError(f"Hugging Face returned HTTP {response.status_code} for {name}.")
                for block in response.iter_bytes(BLOCK_SIZE):
                    received += len(block)
                    if received > size:
                        raise IOError(f"{name} exceeded its expected size.")
                    if received >= next_report:
                        print(f"  {format_bytes(received)} / {format_bytes(size)} read", flush=True)
                        next_report += 512 * MIB
                    yield block
        except httpx.HTTPError as error:
            raise OSError(f"Transfer interrupted for {name} ({type(error).__name__}). Retry the download.") from None
        if received != size:
            raise IOError(f"{name} ended at {received:,} bytes. Expected {size:,} bytes.")

    def copy_video(source, target, size):
        if size <= 0:
            raise ValueError("The archive contains an empty video.")
        check_scratch(size)
        digest = hashlib.md5(usedforsecurity=False)
        written = 0
        with target.open("wb") as output:
            while chunk := source.read(BLOCK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != size:
            raise IOError(f"Extracted {written:,} bytes. Expected {size:,} bytes.")
        return digest.hexdigest()

    class SplitTarReader:
        # tarfile's stream mode needs read(n) to span archive part boundaries.
        def __init__(self, chunks):
            self.chunks = iter(chunks)
            self.pending = memoryview(b"")

        def read(self, size):
            data = bytearray()
            while len(data) < size:
                if not self.pending:
                    self.pending = memoryview(next(self.chunks, b""))
                    if not self.pending:
                        break
                count = min(len(self.pending), size - len(data))
                data.extend(self.pending[:count])
                self.pending = self.pending[count:]
            return bytes(data)

    return SplitTarReader, archive_bytes, check_scratch, copy_video, format_bytes


@app.cell
def _(
    BENCHMARKS,
    SCRATCH_DIR,
    SplitTarReader,
    archive_bytes,
    archive_sizes,
    check_scratch,
    copy_video,
    format_bytes,
    verified_videos,
    zip_name,
):
    def source_key(benchmark):
        spec = BENCHMARKS[benchmark]
        return f'{spec["repo_id"]}@{spec["revision"]}'

    def prepare_dataset(benchmark, paths, sizes, drive, repo_commit):
        spec = BENCHMARKS[benchmark]
        version = source_key(benchmark)
        token = get_token() if benchmark == "StreamBench" else None
        inventory = drive.inventory(paths)
        verified = verified_videos(inventory, version)
        remaining = paths - verified.keys()
        if remaining and benchmark == "StreamBench" and not token:
            raise ValueError(
                "StreamBench is gated. Accept access on its Hugging Face dataset page "
                "and add an approved HF_TOKEN to molab Secrets."
            )
        if remaining:
            if not sizes:
                sizes = archive_sizes(benchmark, paths)
            stage_archives(benchmark, spec, token, sizes, drive, inventory, verified, remaining)

        if remaining:
            examples = ", ".join(str(path) for path in sorted(remaining)[:8])
            raise RuntimeError(f"{len(remaining)} required videos were absent from the archives. Examples: {examples}")

        drive.save_manifest(
            benchmark,
            {
                "benchmark": benchmark,
                "oasis_commit": repo_commit,
                "huggingface_repo": spec["repo_id"],
                "huggingface_revision": spec["revision"],
                "files": {
                    str(path): {"size": int(entry["size"]), "md5": entry["md5Checksum"]}
                    for path, entry in sorted(verified.items())
                },
            },
        )
        return len(verified)

    def stage_archives(benchmark, spec, token, sizes, drive, inventory, verified, remaining):
        capacity = drive.free_bytes()
        if capacity is not None and capacity <= 0 and remaining - inventory.keys():
            raise OSError("Google Drive has no free space for missing videos.")
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        check_scratch(0)
        total = len(verified) + len(remaining)
        with (
            TemporaryDirectory(dir=SCRATCH_DIR, prefix="dataset-") as directory,
            httpx.Client(follow_redirects=True, timeout=60, transport=httpx.HTTPTransport(retries=2)) as client,
            mo.status.progress_bar(total=total, title=f"Preparing {benchmark}") as progress,
        ):
            scratch = Path(directory)
            progress.update(increment=len(verified))

            def deliver(path, source, size):
                nonlocal capacity
                is_new = path not in inventory
                if is_new and capacity is not None and size > capacity:
                    raise OSError(f"Drive has insufficient space for {path} ({format_bytes(size)}).")
                video = scratch / "video.mp4"
                digest = copy_video(source, video, size)
                entry = drive.record_video(path, video, size, digest, source_key(benchmark))
                video.unlink()
                if is_new and capacity is not None:
                    capacity -= size
                verified[path] = entry
                remaining.remove(path)
                progress.update(subtitle=str(path))

            if benchmark == "OVO-Bench":
                # Release parts are consecutive bytes of one uncompressed tar archive.
                def parts():
                    for name, size in sizes.items():
                        yield from archive_bytes(client, spec, name, size, token)

                with closing(parts()) as chunks, tarfile.open(
                    fileobj=SplitTarReader(chunks), mode="r|", bufsize=BLOCK_SIZE
                ) as archive:
                    for member in archive:
                        suffix = PurePosixPath(member.name).parts[-2:]
                        path = PurePosixPath(benchmark, *suffix)
                        if member.isfile() and path in remaining:
                            with archive.extractfile(member) as source:
                                deliver(path, source, member.size)
                        if not remaining:
                            break
                return

            archive_members = {}
            for path in remaining:
                archive_members.setdefault(zip_name(path), set()).add(path)
            for name, targets in sorted(archive_members.items()):
                size = sizes[name]
                check_scratch(size)
                local_zip = scratch / "archive.zip"
                with local_zip.open("wb") as output:
                    for block in archive_bytes(client, spec, name, size, token):
                        output.write(block)

                with zipfile.ZipFile(local_zip) as archive:
                    match_parts = 1 if benchmark == "StreamBench" else 2
                    expected = {path.parts[-match_parts:]: path for path in targets}
                    matched = {}
                    for member in archive.infolist():
                        parts = PurePosixPath(member.filename).parts
                        mode = member.external_attr >> 16
                        if member.is_dir() or stat.S_ISLNK(mode) or "__MACOSX" in parts:
                            continue
                        path = expected.get(parts[-match_parts:])
                        if path is not None:
                            if path in matched:
                                raise ValueError(f"{name} contains multiple members matching {path}.")
                            matched[path] = member
                    if unmatched := targets - matched.keys():
                        examples = ", ".join(str(path) for path in sorted(unmatched)[:5])
                        raise ValueError(f"{name} is missing {len(unmatched)} required videos. Examples: {examples}")
                    for path, member in sorted(matched.items()):
                        with archive.open(member) as source:
                            deliver(path, source, member.file_size)
                local_zip.unlink()

    return prepare_dataset, source_key


@app.cell
def _():
    mo.md(r"""
    # OASIS benchmark setup

    Prepare one benchmark's videos on Google Drive for later evaluation.
    """)
    return


@app.cell
def _():
    drive_form = mo.ui.text(
        value="Thesis-Data/OASIS-testing/datasets",
        label="Drive folder",
        full_width=True,
    ).form(submit_button_label="Connect Google Drive")
    mo.vstack([mo.md("## 1. Google Drive"), drive_form])
    return (drive_form,)


@app.cell
def _(DriveDataset, drive_form, format_bytes):
    mo.stop(drive_form.value is None)
    drive_folder = PurePosixPath(drive_form.value.strip().strip("/"))
    if not drive_folder.parts or ".." in drive_folder.parts or "://" in drive_form.value:
        raise ValueError("Enter a folder path inside My Drive.")
    drive_fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    dataset_drive = DriveDataset(drive_fs, drive_folder)
    dataset_drive.check_access()
    drive_free = dataset_drive.free_bytes()
    _available = format_bytes(drive_free) if drive_free is not None else "unreported"
    mo.md(f"Connected to `{drive_folder}`. Available Drive space: **{_available}**.")
    return (dataset_drive,)


@app.cell
def _():
    clone_button = mo.ui.run_button(label="Clone or reuse OASIS-testing")
    mo.vstack([mo.md("## 2. OASIS repository"), clone_button])
    return (clone_button,)


@app.cell
def _(REPO_DIR, clone_button, clone_oasis):
    mo.stop(not clone_button.value)
    oasis_commit = clone_oasis()
    mo.md(f"Ready at `{REPO_DIR}` · commit `{oasis_commit}`.")
    return (oasis_commit,)


@app.cell
def _(BENCHMARKS):
    benchmark_picker = mo.ui.dropdown(list(BENCHMARKS), value="StreamBench", label="Benchmark")
    mo.vstack(
        [
            mo.md("## 3. Prepare dataset"),
            benchmark_picker,
            mo.md(
                "StreamBench requires [dataset access approval](https://huggingface.co/datasets/Barry-12138/StreamBench_v0.3) "
                "and an `HF_TOKEN` in molab Secrets."
            ),
        ]
    )
    return (benchmark_picker,)


@app.cell
def _(
    BENCHMARKS,
    WORKSPACE,
    archive_sizes,
    benchmark_picker,
    dataset_drive,
    format_bytes,
    oasis_commit,
    required_videos,
    source_key,
    verified_videos,
    zip_name,
):
    selected_benchmark = benchmark_picker.value
    _spec = BENCHMARKS[selected_benchmark]
    video_paths = required_videos(selected_benchmark)
    _inventory = dataset_drive.inventory(video_paths)
    _verified = verified_videos(_inventory, source_key(selected_benchmark))
    _remaining = video_paths - _verified.keys()
    source_sizes = archive_sizes(selected_benchmark, video_paths) if _remaining else {}
    _pending_archives = (
        list(source_sizes)
        if selected_benchmark == "OVO-Bench"
        else sorted({zip_name(path) for path in _remaining})
    )
    _estimate = format_bytes(sum(source_sizes[name] for name in _pending_archives))
    download_button = mo.ui.run_button(label=f"Prepare {selected_benchmark} on Drive")
    mo.vstack(
        [
            mo.md(
                f"""
                Using OASIS commit `{oasis_commit[:12]}` and
                [Hugging Face revision `{_spec['revision'][:12]}`](https://huggingface.co/datasets/{_spec['repo_id']}/tree/{_spec['revision']}).

                - **{len(video_paths):,}** required video paths
                - **{len(_inventory):,}** present on Drive, **{len(_verified):,}** verified for this revision
                - **{_estimate}** estimated maximum Hugging Face transfer across **{len(_pending_archives)}** archive files
                - **{format_bytes(shutil.disk_usage(WORKSPACE).free)}** free on molab's scratch filesystem
                """
            ),
            download_button,
        ]
    )
    return download_button, selected_benchmark, source_sizes, video_paths


@app.cell
def _(dataset_drive, download_button, oasis_commit, prepare_dataset, selected_benchmark, source_sizes, video_paths):
    mo.stop(not download_button.value)
    _count = prepare_dataset(selected_benchmark, video_paths, source_sizes, dataset_drive, oasis_commit)
    mo.md(f"**{selected_benchmark} is ready on Drive.** Verified {_count:,} videos and saved the manifest.")
    return


if __name__ == "__main__":
    app.run()
