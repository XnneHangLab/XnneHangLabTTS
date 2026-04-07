import os

from .models import DownloadStep, DownloadTargetSpec, ManagedPath
from .paths import RuntimePaths

GENIE_BASE_REPO_ID = os.getenv("XH_GENIE_BASE_REPO_ID", "xnnehang/xnnehanglab-geniedata")
FAST_LANGDETECT_LID176_REPO_ID = "xnnehang/fast-langdetect-lid176"
QWEN_TTS_0_6B_REPO_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_TTS_1_7B_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LUMING_GENIE_TTS_REPO_ID = "xnnehang/luming-genie-tts-v2-pro-plus"
LUMING_GSV_LITE_REPO_ID = "xnnehang/luming-gpt-sovits-v2-pro-plus"

GENIE_BASE_REQUIRED_PATHS = [
    "speaker_encoder.onnx",
    "chinese-hubert-base/chinese-hubert-base.onnx",
    "G2P/EnglishG2P/cmudict.rep",
    "G2P/ChineseG2P/opencpop-strict.txt",
    "lid.176.bin",
]
GENIE_BASE_REQUIRED_FILE_PATHS = list(GENIE_BASE_REQUIRED_PATHS)
GENIE_BASE_REQUIRED_DIR_PATHS: list[str] = []

GSV_LITE_REQUIRED_FILE_PATHS = [
    # chinese-hubert-base
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-hubert-base/config.json",
    # chinese-roberta-wwm-ext-large
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    # g2p — OpenJTalk binary dicts (Japanese)
    "g2p/ja/open_jtalk_dic_utf_8-1.11/char.bin",
    "g2p/ja/open_jtalk_dic_utf_8-1.11/matrix.bin",
    # g2p — NLTK averaged perceptron tagger (English)
    "g2p/en/nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json",
    "g2p/en/nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json",
    "g2p/en/nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json",
    # sv — speaker verification model
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "sv/configuration.json",
]
ROBERTA_FILE_PATTERN = [
    "pytorch_model.bin",
    "added_tokens.json",
    "config.json",
    "configuration.json",
    "README.md",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]

QWEN_TTS_REQUIRED_PATHS = [
    "model.safetensors",
    "speech_tokenizer/model.safetensors",
]


def build_managed_paths(paths: RuntimePaths) -> list[ManagedPath]:
    return [
        ManagedPath(key="workspace", label="根目录", path=str(paths.workspace_root)),
        ManagedPath(key="models", label="模型目录", path=str(paths.models_root)),
        ManagedPath(key="genieBase", label="Genie 基础资源", path=str(paths.genie_base_root)),
        ManagedPath(key="downloadLogs", label="下载日志", path=str(paths.download_logs_root)),
    ]


def get_download_target(target_id: str, paths: RuntimePaths) -> DownloadTargetSpec:
    if target_id == "genie-base":
        return DownloadTargetSpec(
            target_id="genie-base",
            label="GenieData 基础资源",
            provider="modelscope",
            repo_id=GENIE_BASE_REPO_ID,
            allow_file_pattern=[],
            local_dir=paths.genie_base_root,
            resource_root=paths.genie_base_root,
            required_paths=GENIE_BASE_REQUIRED_PATHS,
            required_file_paths=GENIE_BASE_REQUIRED_FILE_PATHS,
            required_dir_paths=GENIE_BASE_REQUIRED_DIR_PATHS,
            download_steps=[
                DownloadStep(
                    repo_id=GENIE_BASE_REPO_ID,
                    local_dir=paths.genie_base_root,
                ),
                DownloadStep(
                    repo_id=FAST_LANGDETECT_LID176_REPO_ID,
                    local_dir=paths.genie_base_root,
                    allow_file_pattern=["lid.176.bin"],
                ),
            ],
        )

    if target_id == "gsv-lite":
        root = paths.gsv_lite_root
        return DownloadTargetSpec(
            target_id="gsv-lite",
            label="GSV-Lite 数据包",
            provider="modelscope",
            repo_id="pengzhendong/chinese-hubert-base",  # primary (unused when steps present)
            allow_file_pattern=[],
            local_dir=root,
            resource_root=root,
            required_paths=GSV_LITE_REQUIRED_FILE_PATHS,
            required_file_paths=GSV_LITE_REQUIRED_FILE_PATHS,
            required_dir_paths=[],
            download_steps=[
                DownloadStep(
                    repo_id="pengzhendong/chinese-hubert-base",
                    local_dir=root / "chinese-hubert-base",
                ),
                DownloadStep(
                    repo_id="dienstag/chinese-roberta-wwm-ext-large",
                    local_dir=root / "chinese-roberta-wwm-ext-large",
                    allow_file_pattern=ROBERTA_FILE_PATTERN,
                ),
                DownloadStep(
                    repo_id="xnnehang/gsv-v2proplus-g2p-resource",
                    local_dir=root / "g2p",
                ),
                DownloadStep(
                    repo_id="xnnehang/gsv-v2proplus-sv-resource",
                    local_dir=root / "sv",
                ),
            ],
        )

    if target_id == "qwen-tts-0.6b":
        return DownloadTargetSpec(
            target_id="qwen-tts-0.6b",
            label="Qwen3-TTS 0.6B",
            provider="modelscope",
            repo_id=QWEN_TTS_0_6B_REPO_ID,
            allow_file_pattern=[],
            local_dir=paths.qwen_tts_0_6b_root,
            resource_root=paths.qwen_tts_0_6b_root,
            required_paths=QWEN_TTS_REQUIRED_PATHS,
            required_file_paths=QWEN_TTS_REQUIRED_PATHS,
        )

    if target_id == "qwen-tts-1.7b":
        return DownloadTargetSpec(
            target_id="qwen-tts-1.7b",
            label="Qwen3-TTS 1.7B",
            provider="modelscope",
            repo_id=QWEN_TTS_1_7B_REPO_ID,
            allow_file_pattern=[],
            local_dir=paths.qwen_tts_1_7b_root,
            resource_root=paths.qwen_tts_1_7b_root,
            required_paths=QWEN_TTS_REQUIRED_PATHS,
            required_file_paths=QWEN_TTS_REQUIRED_PATHS,
        )

    if target_id == "luming-genie-tts-v2-pro-plus":
        return DownloadTargetSpec(
            target_id="luming-genie-tts-v2-pro-plus",
            label="鹿鸣 Genie-TTS v2 Pro+",
            provider="modelscope",
            repo_id=LUMING_GENIE_TTS_REPO_ID,
            allow_file_pattern=[],
            local_dir=paths.genie_tts_luming_v2_pro_plus_root,
            resource_root=paths.genie_tts_luming_v2_pro_plus_root,
            required_paths=[],
            required_file_paths=[],
        )

    if target_id == "luming-gsv-lite-v2-pro-plus":
        return DownloadTargetSpec(
            target_id="luming-gsv-lite-v2-pro-plus",
            label="鹿鸣 GSV-Lite v2 Pro+",
            provider="modelscope",
            repo_id=LUMING_GSV_LITE_REPO_ID,
            allow_file_pattern=[],
            local_dir=paths.gsv_tts_lite_luming_v2_pro_plus_root,
            resource_root=paths.gsv_tts_lite_luming_v2_pro_plus_root,
            required_paths=[],
            required_file_paths=[],
        )

    raise KeyError(f"unsupported target: {target_id}")
