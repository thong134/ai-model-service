from __future__ import annotations

import argparse
import logging
import random
import string
from pathlib import Path
from typing import Iterable, List

import pandas as pd


LABEL_NORMALIZATION = {
    "non_toxic": "neutral",
    "clean": "neutral",
    "ham": "neutral",
    "offensive": "toxic",
    "abusive": "toxic",
}

POSITIVE_PREFIXES = [
    "",
    "Thật sự mà nói,",
    "Cá nhân mình thấy",
    "Theo trải nghiệm của mình,",
    "Mình phải công nhận",
    "Bạn bè mình cũng đồng ý là",
    "Nhìn chung,",
    "Thú thật là",
]
POSITIVE_SUBJECTS = [
    "Dịch vụ",
    "Trải nghiệm",
    "Sản phẩm",
    "Không gian",
    "Đội ngũ nhân viên",
    "Phong cách phục vụ",
    "Hương vị",
    "Chất lượng",
    "Ứng dụng",
    "Bài viết",
]
POSITIVE_QUALIFIERS = [
    "quá tuyệt vời",
    "khiến tôi hài lòng",
    "thực sự đỉnh cao",
    "luôn vượt mong đợi",
    "khiến tôi muốn quay lại",
    "đáng giá từng đồng",
    "khiến cả gia đình thích mê",
    "khiến mình cười cả ngày",
    "tạo cảm hứng tích cực",
    "làm mình thấy tự hào",
]
POSITIVE_ENDINGS = [
    "!",
    " vô cùng.",
    " và chắc chắn sẽ giới thiệu cho bạn bè.",
    ", cảm ơn rất nhiều!",
    ", tuyệt vời lắm luôn.",
    ", đúng chuẩn 5 sao.",
    ", mong được ủng hộ dài lâu.",
]

NEGATIVE_PREFIXES = [
    "",
    "Nói thật,",
    "Cảm giác của mình là",
    "Thành thật mà nói,",
    "Theo mình thấy",
    "Cả nhóm đều thấy",
    "Riêng mình nghĩ",
    "Xin góp ý rằng",
]
NEGATIVE_SUBJECTS = [
    "Dịch vụ",
    "Trải nghiệm",
    "Chất lượng",
    "Thái độ nhân viên",
    "Sản phẩm",
    "Bài viết",
    "Ứng dụng",
    "Giao hàng",
    "Âm thanh",
    "Món ăn",
]
NEGATIVE_QUALIFIERS = [
    "khiến tôi thất vọng",
    "thực sự quá tệ",
    "không đáng tiền",
    "làm mình mất thời gian",
    "khiến cả nhóm bực bội",
    "kém xa kỳ vọng",
    "rất thiếu chuyên nghiệp",
    "khiến mình bực cả ngày",
    "không muốn quay lại nữa",
    "cần xem lại ngay",
]
NEGATIVE_ENDINGS = [
    ".",
    " và chắc chắn không quay lại.",
    ", mong sớm cải thiện.",
    ", tệ chưa từng thấy.",
    ", làm mình tức điên.",
    ", cảm giác như bị lừa.",
]

SPAM_ACTIONS = [
    "Click",
    "Nhấn",
    "Truy cập",
    "Tham gia",
    "Đăng ký",
    "Xem",
]
SPAM_CALLS = [
    "link siêu ưu đãi",
    "deal hot cuối tuần",
    "khuyến mãi 90%",
    "voucher miễn phí",
    "quà tặng cực lớn",
    "combo mua 1 tặng 5",
]
SPAM_ENDINGS = [
    " trước khi hết chỗ!",
    " nhận quà liền tay!",
    " số lượng có hạn!",
    " đừng bỏ lỡ!",
    " để được hoàn tiền 100%!",
    " bảo đảm lợi nhuận 300%!",
]

NEUTRAL_TEMPLATES = [
    "Thông tin khá hữu ích, cảm ơn đã chia sẻ.",
    "Mình sẽ suy nghĩ thêm trước khi quyết định.",
    "Đọc xong thấy cũng bình thường thôi.",
    "Nội dung dễ hiểu và rõ ràng.",
    "Chưa biết nên nhận xét thế nào, đành chờ thêm.",
    "Khá ổn, không có gì nổi bật.",
    "Thông tin trung lập, phù hợp để tham khảo.",
    "Tạm được, cảm ơn bạn.",
    "Ghi nhận ý kiến của bạn.",
    "Cũng đáng để cân nhắc thử.",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    return LABEL_NORMALIZATION.get(normalized, normalized)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'comment' and 'label' columns")
    df = df.dropna(subset=["comment", "label"])
    df["comment"] = df["comment"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).apply(normalize_label)
    df = df[df["comment"] != ""]
    df = df.drop_duplicates(subset="comment")
    return df


def batch_generate(sentences: Iterable[str], label: str) -> pd.DataFrame:
    return pd.DataFrame({"comment": list(sentences), "label": label})


def build_sentences(
    subjects: List[str],
    qualifiers: List[str],
    endings: List[str],
    prefixes: List[str],
    count: int,
) -> List[str]:
    generated: set[str] = set()
    sentences: List[str] = []
    attempts = 0
    while len(sentences) < count and attempts < count * 10:
        attempts += 1
        subject = random.choice(subjects)
        qualifier = random.choice(qualifiers)
        ending = random.choice(endings)
        prefix = random.choice(prefixes)
        if prefix:
            sentence = f"{prefix} {subject.lower()} {qualifier}{ending}".strip()
        else:
            sentence = f"{subject} {qualifier}{ending}"
        sentence = sentence[0].upper() + sentence[1:]
        if sentence in generated:
            continue
        generated.add(sentence)
        sentences.append(sentence)
    while len(sentences) < count:
        base = random.choice(sentences) if sentences else "Nội dung khá ổn."
        variation = f"{base} ({len(sentences) + 1})"
        if variation in generated:
            variation = f"{base} #{len(sentences) + 1}"
        generated.add(variation)
        sentences.append(variation)
    return sentences


def random_promo_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def generate_spam(count: int) -> List[str]:
    sentences: List[str] = []
    for _ in range(count):
        action = random.choice(SPAM_ACTIONS)
        call = random.choice(SPAM_CALLS)
        ending = random.choice(SPAM_ENDINGS)
        domain = f"http://deal{random.randint(100, 999)}.vn/{random_promo_code()}"
        sentence = f"{action} {call} tại {domain}{ending}"
        sentences.append(sentence)
    return sentences


def generate_neutral(count: int) -> List[str]:
    sentences: List[str] = []
    for _ in range(count):
        sentences.append(random.choice(NEUTRAL_TEMPLATES))
    return sentences


def augment_dataset(
    df: pd.DataFrame,
    positive_count: int,
    negative_count: int,
    spam_count: int,
    neutral_sample: int,
    toxic_extra: int,
) -> pd.DataFrame:
    random.seed(42)

    df["label"] = df["label"].apply(normalize_label)

    neutral_df = df[df["label"] == "neutral"].copy()
    if not neutral_df.empty:
        sample_size = min(neutral_sample, len(neutral_df))
        sampled_neutral = neutral_df.sample(n=sample_size, random_state=42, replace=sample_size > len(neutral_df))
        synthetic_neutral = generate_neutral(max(0, neutral_sample - sample_size))
        neutral_frames = [sampled_neutral]
        if synthetic_neutral:
            neutral_frames.append(batch_generate(synthetic_neutral, "neutral"))
        neutral_augmented = pd.concat(neutral_frames, ignore_index=True)
    else:
        neutral_augmented = batch_generate(generate_neutral(neutral_sample), "neutral")

    positive_sentences = build_sentences(
        POSITIVE_SUBJECTS,
        POSITIVE_QUALIFIERS,
        POSITIVE_ENDINGS,
        POSITIVE_PREFIXES,
        positive_count,
    )
    negative_sentences = build_sentences(
        NEGATIVE_SUBJECTS,
        NEGATIVE_QUALIFIERS,
        NEGATIVE_ENDINGS,
        NEGATIVE_PREFIXES,
        negative_count,
    )
    spam_sentences = generate_spam(spam_count)

    toxic_sentences: List[str] = []
    if toxic_extra > 0:
        toxic_prefix = [
            "Mày",
            "Đồ",
            "Thể loại",
            "Cái thứ",
            "Loại người",
        ]
        toxic_suffix = [
            "đáng bị block",
            "làm ô nhiễm mạng xã hội",
            "chỉ giỏi khẩu nghiệp",
            "khiến ai cũng ghét",
            "ngu không ai cứu nổi",
        ]
        toxic_suffix_end = [
            "!",
            ", cút đi!",
            ", biến cho nhanh.",
            ", đọc mà tức.",
            ", làm người ta phát bực.",
        ]
        for _ in range(toxic_extra):
            prefix = random.choice(toxic_prefix)
            suffix = random.choice(toxic_suffix)
            ending = random.choice(toxic_suffix_end)
            toxic_sentences.append(f"{prefix} {suffix}{ending}")

    augmented_frames = [
        df,
        neutral_augmented,
        batch_generate(positive_sentences, "positive"),
        batch_generate(negative_sentences, "negative"),
        batch_generate(spam_sentences, "spam"),
    ]

    if toxic_sentences:
        augmented_frames.append(batch_generate(toxic_sentences, "toxic"))

    augmented_df = pd.concat(augmented_frames, ignore_index=True)
    augmented_df = augmented_df.drop_duplicates(subset="comment").reset_index(drop=True)

    return augmented_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment Vietnamese review moderation dataset")
    parser.add_argument("input", type=Path, help="Path to the input CSV dataset")
    parser.add_argument("output", type=Path, help="Path to save the augmented dataset")
    parser.add_argument("--positive-count", type=int, default=2000, help="Number of synthetic positive samples")
    parser.add_argument("--negative-count", type=int, default=2000, help="Number of synthetic negative samples")
    parser.add_argument("--spam-count", type=int, default=2000, help="Number of synthetic spam samples")
    parser.add_argument("--neutral-sample", type=int, default=2000, help="Number of neutral samples to retain/generate")
    parser.add_argument("--toxic-extra", type=int, default=1000, help="Additional synthetic toxic samples")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("augment")

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    logger.info("Loading dataset from %s", input_path)
    df = load_dataset(input_path)

    logger.info("Original label distribution: %s", df["label"].value_counts().to_dict())

    augmented_df = augment_dataset(
        df,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
        spam_count=args.spam_count,
        neutral_sample=args.neutral_sample,
        toxic_extra=args.toxic_extra,
    )

    logger.info("Augmented label distribution: %s", augmented_df["label"].value_counts().to_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    logger.info("Augmented dataset saved to %s (%d rows)", output_path, len(augmented_df))


if __name__ == "__main__":
    main()
