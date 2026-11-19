# OASIS molab setup plan

The first version of [OASIS-molab.py](OASIS-molab.py) will connect to Google Drive, clone OASIS-testing, and prepare the videos for one selected benchmark. The milestone is a verified copy of the required videos on Drive. Model installation and evaluation are later steps.

## 1. Connect Google Drive

**Decision.** Use molab's Google Drive connection through `gdrive-fsspec`, following the existing CogStream notebook. Make the destination configurable and propose `Thesis-Data/OASIS-testing/datasets` as the default. Verify access with a small write and read-back, then check for an existing dataset copy and available Drive capacity before starting a transfer. Store any required tokens through molab Secrets. [Marimo remote storage guide][remote-storage] · [CogStream notebook](../CogStream-testing/CogStream-molab.py#L32) · [Molab storage and secrets][molab-storage]

**Rationale.** Drive gives the videos and transfer progress a home across sessions. Even when a large download fits during a session, molab's current policy excludes files larger than 1 GB from teardown sync, and ordinary workspace data is treated as scratch. Marimo documents sessions of up to 12 hours. No numerical live session disk quota was found in the public docs, and available space has yet to be measured in a running molab session. Check free space on the actual staging filesystem at launch and again before each archive, accounting for the archive, temporary extraction, and any download cache. [Molab storage policy][molab-storage] · [Molab compute and session limits][molab-compute]

## 2. Clone the test repository

**Decision.** Clone [chillCode404/OASIS-testing](https://github.com/chillCode404/OASIS-testing.git) over HTTPS into a dedicated local workspace directory. If a checkout is already there, verify its origin, reuse it, and show its commit. Verify that the selected benchmark's metadata is present. HTTPS access from molab still needs a live test.

**Rationale.** The checked-out commit fixes the metadata and evaluator code that define the dataset layout. Showing the commit makes it possible to reproduce a run and explain any change in the required video list. Keeping the clone local gives Git a normal filesystem and keeps large data out of the repository. [OASIS-testing dataset and evaluation instructions](README.md#datasets) · [Dataset ignore rule](.gitignore#L10)

## 3. Prepare one selected benchmark

**Decision.** Use a single-select dropdown and an explicit download button. On selection, show the source revision, estimated download size, required video count, and what is already on Drive. Start with StreamBench to exercise the connection, download, layout, and upload with the smallest release. Drive copies will preserve each relative path under the dataset root. [StreamBench release][streambench-files] · [OASIS-testing dataset layout](README.md#datasets)

**Rationale.** The bundled OASIS metadata defines the files this checkout can evaluate. For each row, the evaluator joins `info.video_path` to `--dataset_root` and skips the row if that local file is missing. Use the unique paths from the selected metadata as the download checklist. A later evaluation stage can stage the needed Drive files onto the local filesystem in that same layout. [Evaluator path check](src/scripts/eval.py#L164)

### OVO-Bench

- **Why this subset.** The OASIS paper reports Real-Time Visual Perception and Backward Tracing. OVO-Bench also has a Forward Active Responding track. The OASIS checkout's `eval_ovo_bench.json` contains **1,468 unique video paths**, all under `OVO-Bench/chunked_videos/`. The official release offers both raw source videos and pre-chunked videos. The OASIS README identifies the pre-chunked clips as the expected input. Using the raw source release would require reproducing and verifying the clip preprocessing. [OASIS paper, §4.1, Table 1 and §6.1][oasis-paper] · [OVO-Bench data preparation][ovo-upstream] · [OASIS dataset layout](README.md#datasets) · [OASIS OVO-Bench metadata](metadata/eval_ovo_bench.json)
- **Download decision.** Use only the `chunked_videos.tar.partaa` through `partao` files, about **153 GB** of the approximately **200 GB** Hugging Face release. Read the parts in order as one tar stream. Stage one matching video at a time, upload it, and skip tar entries outside the metadata checklist. All tar parts may still need to be read because the release packages the clips as a split archive. This saves the raw-source download and limits local disk usage even when the stream is read end to end. Verify archive member names against the checklist before treating this path as proven. [OVO-Bench release files][ovo-files] · [OVO-Bench data preparation][ovo-upstream]

### StreamingBench

- **Why this subset.** OASIS Table 2 reports Real-Time Visual Understanding, Misleading Context Understanding (MCU), Anomaly Context Understanding (ACU), and Sequential Question Answering (SQA). The checkout has metadata for these four tasks, covering **650 unique paths**. The published release also contains archives for Emotion Recognition, Multimodal Alignment, Scene Understanding, Source Discrimination, and Proactive Output. Those archives have no matching paths in the bundled metadata. The paper's **All Context** setting describes the video context available for each included case. Table 2 identifies which tasks are included. [OASIS paper, §4.1, Table 2 and §6.2][oasis-paper] · [StreamingBench evaluation settings][streamingbench-upstream] · [StreamingBench release files][streamingbench-files] · [OASIS StreamingBench metadata](metadata/StreamingBench)
- **Download decision.** Use the ten Real-Time Visual Understanding ZIPs (500 paths), the MCU ZIP (50), the ACU ZIP (50), and the two SQA ZIPs (50). Their listed sizes add up to about **160 GB** of the approximately **203 GB** release. Process one ZIP at a time and keep only the video members the metadata references. Start with ACU (9.18 GB) as a layout check before processing the other archives. Use `info.video_path` to resolve naming differences such as `Misleading_Context_Recognition.json` pointing to the directory `Misleading_Context_Understanding/`. [StreamingBench release files][streamingbench-files] · [OASIS MCU metadata](metadata/StreamingBench/Misleading_Context_Recognition.json)

### StreamBench

- **Download decision.** Use `Ego.zip`, `Movie.zip`, and `WebVideo.zip`, about **8.4 GB** total. Process each archive in turn and preserve the `StreamBench/{Ego,Movie,WebVideo}/` paths. The bundled metadata has 307 rows referring to **275 unique video paths**. The Hugging Face account must accept the dataset's access conditions before its read token can download the archives. [StreamBench release files][streambench-files] · [Hugging Face gated dataset access][gated-datasets] · [OASIS StreamBench metadata](metadata/eval_streambench.json)

## Completion and recovery

Record the OASIS commit, Hugging Face revision, and verified uploads in a small Drive manifest. Check each upload against the extracted file size and use a checksum when available. Clear local staging data after verification. If a session ends, skip verified files on the next run. For OVO-Bench, a retry may have to replay the tar stream to reach the next missing member. If an individual ZIP exceeds the measured scratch space, use an existing Drive copy or stage that archive on a machine with sufficient disk. [Hugging Face repository file metadata][hf-files-api] · [Molab session limits][molab-compute]

Before declaring the notebook ready, run `marimo check`, then test Drive authentication, the clone, and the disk check in a live molab session. Use StreamBench for the first end-to-end transfer and check that rerunning reuses verified uploads. Mark any benchmark complete only when every required relative path has a verified video on Drive.

[oasis-paper]: https://arxiv.org/pdf/2604.17052v1
[ovo-upstream]: https://github.com/JoeLeelyf/OVO-Bench#data-preparation
[ovo-files]: https://huggingface.co/datasets/JoeLeelyf/OVO-Bench/tree/main
[streamingbench-upstream]: https://github.com/THUNLP-MT/StreamingBench#performance-of-various-mllms-on-streamingbench
[streamingbench-files]: https://huggingface.co/datasets/mjuicem/StreamingBench/tree/main
[streambench-files]: https://huggingface.co/datasets/Barry-12138/StreamBench_v0.3/tree/main
[remote-storage]: https://docs.marimo.io/guides/working_with_data/remote_storage/#fsspec
[molab-storage]: https://marimo.io/pages/molab/storage
[molab-compute]: https://docs.marimo.io/guides/molab/#compute
[gated-datasets]: https://huggingface.co/docs/hub/en/datasets-gated
[hf-files-api]: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree
