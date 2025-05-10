# It does sampling of messages with context from the relevant channels in multi-step
import os
import pickle
import random
from pathlib import Path

# Import db_utilites
# has functions that query the db (get records)
import db_utilities
import pandas as pd
from tqdm import tqdm

# --- Configuration ---

# MongoDB Configuration
DB_NAME = "Telegram_test"

# File Paths & Cache Directory
BASE_DIR = Path(".")
LABELED_DATA_DIR = BASE_DIR / "labeled_data"
CACHE_DIR = BASE_DIR / "cache"  # Directory to store cache files
LANGUAGE_FILE = LABELED_DATA_DIR / "channel_to_language_mapping.csv"
TOPIC_FILE = LABELED_DATA_DIR / "ch_to_topic_mapping.csv"
CONSPIRACY_FILE = LABELED_DATA_DIR / "conspiracy_channels.csv"

# Cache files
CACHE_CHANNEL_SETS_FILE = CACHE_DIR / "channel_sets.pkl"
CACHE_RELEVANT_POOL_FILE = CACHE_DIR / "relevant_message_pool.pkl"
CACHE_BENIGN_POOL_FILE = CACHE_DIR / "benign_message_pool.pkl"

# Sampling Parameters
TOTAL_MESSAGES_TO_ANNOTATE = 5000
PROPORTION_BENIGN = 0.50
N_BENIGN_SAMPLES = int(TOTAL_MESSAGES_TO_ANNOTATE * PROPORTION_BENIGN)
N_RELEVANT_SAMPLES = TOTAL_MESSAGES_TO_ANNOTATE - N_BENIGN_SAMPLES
MAX_BENIGN_CHANNELS_TO_PROCESS = 50

CONTEXT_WINDOW_SIZE = 2
MIN_MESSAGE_LENGTH = 10

RANDOM_SEED = 42
OUTPUT_CSV_FILE = BASE_DIR / "sampled_messages_for_annotation_with_context.csv"

# Filtering Criteria
TARGET_LANGUAGE = "en"
# If the channel falls under these topics accr to their existing label
RELEVANT_TOPICS = [
    "Carding",
    "Extremists and radicals",
    "Porn",
    "Crypto",
    "Software",
]
# If the channel description contains these keywords

CHANNEL_KEYWORDS = [
    "hacker",
    "crack",
    "phish",
    "malware",
    "botnet",
    "exploit",
    "fraud",
    "scam",
    "carding",
    "dump",
    "porn",
    "hate speech",
    "racism",
    "copyright",
    "piracy",
    "cyber warfare",
    "leak",
    "breach",
    "terrorist",
    "weapon",
    "drug",
    "dark web",
    "ransomware",
]
# If the message contains these keywords
MESSAGE_KEYWORDS = [
    "hack",
    "crack",
    "phish",
    "phishing",
    "malware",
    "virus",
    "trojan",
    "botnet",
    "ddos",
    "dos",
    "exploit",
    "vulnerability",
    "forgery",
    "fraud",
    "scam",
    "identity theft",
    "carding",
    "bin",
    "dump",
    "ccv",
    "credit card",
    "cvv",
    "cc",
    "porn",
    "child porn",
    "cp",
    "hate speech",
    "racism",
    "extremist",
    "nazi",
    "radical",
    "spam",
    "unsolicited",
    "copyright",
    "piracy",
    "infringement",
    "trademark",
    "counterfeit",
    "launder",
    "money laundering",
    "cyber warfare",
    "espionage",
    "leak",
    "breach",
    "terrorist",
    "terrorism",
    "weapon",
    "drug",
    "dark web",
    "tor",
    "illegal",
    "ransomware",
    "stolen",
    "buy",
    "sell",
    "accounts",
    "logs",
    "combo list",
    "exploit kit",
    "zero day",
    "0day",
    "attack",
    "compromised",
    "infected",
    "payload",
    "keylogger",
    "rootkit",
    "backdoor",
    "anonymity",
    "vpn",
    "proxy",
    "insult",
    "threaten",
    "harass",
    "dox",
    "exposed",
    "fake",
    "deepfake",
    "gambling",
    "betting",
    "odds",
    "casino",
    "slot",
    "invest",
    "crypto scam",
    "pump and dump",
    "rug pull",
    "guns",
    "ammo",
    "explosives",
    "rich",
    "free",
]


def load_csv_mapping(filepath, id_col, value_col, delimiter=","):
    """Loads a CSV into a dictionary mapping."""
    try:
        df = pd.read_csv(
            filepath, delimiter=delimiter, dtype={id_col: int}, low_memory=False
        )
        df.columns = df.columns.str.strip().str.lower()
        id_col_clean = id_col.strip().lower()
        value_col_clean = value_col.strip().lower()
        if id_col_clean not in df.columns or value_col_clean not in df.columns:
            raise ValueError(
                f"Required columns '{id_col}' or '{value_col}' not found in {filepath}"
            )
        mapping = df.set_index(id_col_clean)[value_col_clean].to_dict()
        return mapping
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def load_csv_set(filepath, id_col, delimiter=","):
    """Loads a CSV column into a set."""
    try:
        df = pd.read_csv(
            filepath, delimiter=delimiter, dtype={id_col: int}, low_memory=False
        )
        df.columns = df.columns.str.strip().str.lower()
        id_col_clean = id_col.strip().lower()
        if id_col_clean not in df.columns:
            raise ValueError(f"Required column '{id_col}' not found in {filepath}")
        id_set = set(df[id_col_clean].unique())
        return id_set
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return set()


def contains_keywords(text, keywords):
    """Checks if text contains any of the keywords (case-insensitive)."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def get_message_context(channel_messages_sorted, target_msg_index, window_size):
    """
    Retrieves context messages around a target message index.
    Assumes channel_messages_sorted is a list of message dicts sorted by date.
    Filters out empty strings before joining context.
    Uses the 'message' key for text content.
    """
    prev_context_list = []
    next_context_list = []

    # Get previous messages text
    start_index = max(0, target_msg_index - window_size)
    for i in range(start_index, target_msg_index):
        msg_text = channel_messages_sorted[i].get("message")
        if isinstance(msg_text, str) and msg_text.strip():
            prev_context_list.append(msg_text)

    # Get next messages text
    end_index = min(len(channel_messages_sorted), target_msg_index + window_size + 1)
    for i in range(target_msg_index + 1, end_index):
        msg_text = channel_messages_sorted[i].get("message")
        if isinstance(msg_text, str) and msg_text.strip():
            next_context_list.append(msg_text)

    prev_context_str = " [SEP] ".join(prev_context_list)
    next_context_str = " [SEP] ".join(next_context_list)

    return prev_context_str, next_context_str


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    print("Starting data sampling process with context and caching...")

    # Create cache dir if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using cache : {CACHE_DIR}")

    print(
        f"Targeting {TOTAL_MESSAGES_TO_ANNOTATE} total samples ({N_RELEVANT_SAMPLES} relevant, {N_BENIGN_SAMPLES} benign)."
    )

    # Identify Channel Sets
    print("\nPhase 1: Identifying Channel Sets")

    relevant_english_channels = None
    benign_english_channel_ids = None
    english_channel_ids = None

    # Loading from cache first to avoid huge waiting times
    if CACHE_CHANNEL_SETS_FILE.exists():
        try:
            with open(CACHE_CHANNEL_SETS_FILE, "rb") as f:
                print(f"Loading channel sets from cache: {CACHE_CHANNEL_SETS_FILE}...")
                cached_data = pickle.load(f)
                relevant_english_channels = cached_data["relevant"]
                benign_english_channel_ids = cached_data["benign"]
                english_channel_ids = cached_data["all_english"]
                print("Channel sets loaded from cache successfully.")
        except Exception as e:
            print(
                f"Warning: Failed to load channel sets from cache ({e}). Recomputing..."
            )
            relevant_english_channels = None

    # If cache loading failed or file doesn't exist, compute = expensive
    if relevant_english_channels is None:
        print("Computing channel sets (cache miss or load failure)...")
        print("Loading auxiliary data...")
        lang_map = load_csv_mapping(
            LANGUAGE_FILE, id_col="ch_id", value_col="language", delimiter="\t"
        )
        topic_map = load_csv_mapping(
            TOPIC_FILE, id_col="ch_ID", value_col="topic", delimiter=","
        )
        conspiracy_set = load_csv_set(CONSPIRACY_FILE, id_col="ch_id", delimiter=",")

        # Identify Relevant and Benign English Channels
        print("Classifying all channels...")
        all_channel_ids = db_utilities.get_channel_ids(DB_NAME)

        relevant_english_channels = {}
        benign_english_channel_ids = set()
        english_channel_ids = set()

        channel_metadata_cursor = db_utilities.MongoClient(db_utilities.uri)[
            DB_NAME
        ].Channel.find({}, {"_id": 1, "title": 1, "description": 1, "scam": 1})

        for channel_meta in tqdm(
            channel_metadata_cursor,
            total=len(all_channel_ids),
            desc="Classifying Channels",
        ):
            ch_id = channel_meta["_id"]
            if lang_map.get(ch_id) == TARGET_LANGUAGE:
                english_channel_ids.add(ch_id)
                reasons = []
                if topic_map.get(ch_id) in RELEVANT_TOPICS:
                    reasons.append(f"Topic:{topic_map.get(ch_id)}")
                title = channel_meta.get("title", "")
                description = channel_meta.get("description", "")
                if contains_keywords(title, CHANNEL_KEYWORDS) or contains_keywords(
                    description, CHANNEL_KEYWORDS
                ):
                    reasons.append("ChannelKeyword")
                if channel_meta.get("scam", False):
                    reasons.append("ScamFlag")
                # Just to get insight if annotater disagree (maybe helps?) if need be...
                if reasons:
                    relevant_english_channels[ch_id] = ";".join(reasons)
                else:
                    benign_english_channel_ids.add(ch_id)

        # Save computed sets to cache to avoid computations in future for same
        try:
            print(
                f"Saving computed channel sets to cache: {CACHE_CHANNEL_SETS_FILE}..."
            )
            cache_data = {
                "relevant": relevant_english_channels,
                "benign": benign_english_channel_ids,
                "all_english": english_channel_ids,
            }
            with open(CACHE_CHANNEL_SETS_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            print("Channel sets saved to cache.")
        except Exception as e:
            print(f"Warning: Failed to save channel sets to cache ({e}).")

    # Print loaded/processed data
    print(f"Total English channels: {len(english_channel_ids)}")
    print(f"Potentially relevant English channels: {len(relevant_english_channels)}")
    print(f"Potentially benign English channels: {len(benign_english_channel_ids)}")

    # Sampling Relevant Messages
    print(f"\nSampling {N_RELEVANT_SAMPLES} Relevant Messages")

    sampled_relevant_messages = []
    relevant_message_pool = None

    # Try loading relevant pool from cache
    if N_RELEVANT_SAMPLES > 0 and CACHE_RELEVANT_POOL_FILE.exists():
        try:
            with open(CACHE_RELEVANT_POOL_FILE, "rb") as f:
                print(
                    f"Loading relevant message pool from cache: {CACHE_RELEVANT_POOL_FILE}"
                )
                relevant_message_pool = pickle.load(f)
                print(
                    f"Relevant message pool loaded from cache ({len(relevant_message_pool)} messages)."
                )
        except Exception as e:
            print(
                f"Warning: Failed to load relevant pool from cache ({e}). Recomputing..."
            )
            relevant_message_pool = None

    # Compute relevant pool if needed
    if N_RELEVANT_SAMPLES > 0 and relevant_message_pool is None:
        if not relevant_english_channels:
            print(
                "Warning: No relevant channels identified. Cannot build relevant pool."
            )
            relevant_message_pool = []
        else:
            print("Building relevant message pool (cache miss or load failure)...")
            relevant_message_pool = []
            for ch_id, reason_str in tqdm(
                relevant_english_channels.items(), desc="Processing Relevant Channels"
            ):
                try:
                    messages_dict = db_utilities.get_text_messages_by_id_ch(
                        ch_id, DB_NAME
                    )
                    channel_messages_list = []
                    for msg_id, msg_details in messages_dict.items():
                        if "message" in msg_details and isinstance(
                            msg_details["message"], str
                        ):
                            msg_details["message_id"] = msg_id
                            channel_messages_list.append(msg_details)
                    channel_messages_list.sort(key=lambda x: x.get("date", 0) or 0)
                    for idx, msg_details in enumerate(channel_messages_list):
                        msg_text = msg_details["message"]
                        if len(msg_text) < MIN_MESSAGE_LENGTH:
                            continue
                        if not contains_keywords(msg_text, MESSAGE_KEYWORDS):
                            continue

                        prev_ctx, next_ctx = get_message_context(
                            channel_messages_list, idx, CONTEXT_WINDOW_SIZE
                        )

                        relevant_message_pool.append(
                            {
                                "message_id": msg_details["message_id"],
                                "channel_id": ch_id,
                                "date": msg_details.get("date"),
                                "text": msg_text,
                                "previous_context": prev_ctx,
                                "next_context": next_ctx,
                                "selection_reason": f"RelevantChannel({reason_str});MessageKeywordMatch",
                            }
                        )
                except Exception as e:
                    pass

            # Save computed pool to cache
            try:
                print(
                    f"Saving computed relevant pool ({len(relevant_message_pool)} messages) to cache: {CACHE_RELEVANT_POOL_FILE}..."
                )
                with open(CACHE_RELEVANT_POOL_FILE, "wb") as f:
                    pickle.dump(relevant_message_pool, f)
                print("Relevant pool saved to cache.")
            except Exception as e:
                print(f"Warning: Failed to save relevant pool to cache ({e}).")

    # Sample from the (loaded or computed) relevant pool
    if N_RELEVANT_SAMPLES > 0:
        if not relevant_message_pool:
            print(
                "Warning: Relevant message pool is empty. No relevant sampling possible."
            )
        elif len(relevant_message_pool) <= N_RELEVANT_SAMPLES:
            print(
                f"Warning: Relevant pool size ({len(relevant_message_pool)}) is <= target ({N_RELEVANT_SAMPLES}). Taking all."
            )
            sampled_relevant_messages = relevant_message_pool
        else:
            print(
                f"Sampling {N_RELEVANT_SAMPLES} from relevant pool of {len(relevant_message_pool)}..."
            )
            sampled_relevant_messages = random.sample(
                relevant_message_pool, N_RELEVANT_SAMPLES
            )
        print(f"Sampled {len(sampled_relevant_messages)} relevant messages.")
    else:
        print("Skipping relevant message sampling.")

    # Sampling Benign Messages
    print(f"\n--- Phase 3: Sampling {N_BENIGN_SAMPLES} Benign Messages ---")

    sampled_benign_messages = []
    benign_message_pool = None

    # Try loading benign pool from cache
    if N_BENIGN_SAMPLES > 0 and CACHE_BENIGN_POOL_FILE.exists():
        try:
            with open(CACHE_BENIGN_POOL_FILE, "rb") as f:
                print(
                    f"Loading benign message pool from cache: {CACHE_BENIGN_POOL_FILE}"
                )
                benign_message_pool = pickle.load(f)
                print(
                    f"Benign message pool loaded from cache ({len(benign_message_pool)} messages)."
                )
        except Exception as e:
            print(f"Failed to load benign pool from cache: ({e})")
            benign_message_pool = None

    # Compute benign pool if needed
    if N_BENIGN_SAMPLES > 0 and benign_message_pool is None:
        if not benign_english_channel_ids:
            print("Warning: No benign channels identified. Cannot build benign pool.")
            benign_message_pool = []
        else:
            print("Building benign message pool (cache miss or load failure)")
            benign_message_pool = []

            channels_to_process = random.sample(
                list(benign_english_channel_ids),
                min(len(benign_english_channel_ids), MAX_BENIGN_CHANNELS_TO_PROCESS),
            )
            print(
                f"Processing up to {len(channels_to_process)} benign channels...will take a while"
            )

            for ch_id in tqdm(channels_to_process, desc="Processing Benign Channels"):
                try:
                    messages_dict = db_utilities.get_text_messages_by_id_ch(
                        ch_id, DB_NAME
                    )
                    channel_messages_list = []
                    for msg_id, msg_details in messages_dict.items():
                        if "message" in msg_details and isinstance(
                            msg_details["message"], str
                        ):
                            msg_details["message_id"] = msg_id
                            channel_messages_list.append(msg_details)
                    channel_messages_list.sort(key=lambda x: x.get("date", 0) or 0)
                    for idx, msg_details in enumerate(channel_messages_list):
                        msg_text = msg_details["message"]
                        if len(msg_text) < MIN_MESSAGE_LENGTH:
                            continue
                        prev_ctx, next_ctx = get_message_context(
                            channel_messages_list, idx, CONTEXT_WINDOW_SIZE
                        )
                        benign_message_pool.append(
                            {
                                "message_id": msg_details["message_id"],
                                "channel_id": ch_id,
                                "date": msg_details.get("date"),
                                "text": msg_text,
                                "previous_context": prev_ctx,
                                "next_context": next_ctx,
                                "selection_reason": "BenignChannelSample",
                            }
                        )
                except Exception as e:
                    pass

            # Save computed pool to cache
            try:
                print(
                    f"Saving computed benign pool ({len(benign_message_pool)} messages) to cache: {CACHE_BENIGN_POOL_FILE}..."
                )
                with open(CACHE_BENIGN_POOL_FILE, "wb") as f:
                    pickle.dump(benign_message_pool, f)
                print("Benign pool saved to cache.")
            except Exception as e:
                print(f"Warning: Failed to save benign pool to cache ({e}).")

    # Sample from the (loaded or computed) benign pool
    if N_BENIGN_SAMPLES > 0:
        if not benign_message_pool:
            print("Warning: Benign message pool is empty. No benign sampling possible.")
        elif len(benign_message_pool) <= N_BENIGN_SAMPLES:
            print(
                f"Warning: Benign pool size ({len(benign_message_pool)}) is <= target ({N_BENIGN_SAMPLES}). Taking all."
            )
            sampled_benign_messages = benign_message_pool
        else:
            print(
                f"Sampling {N_BENIGN_SAMPLES} from benign pool of {len(benign_message_pool)}..."
            )
            sampled_benign_messages = random.sample(
                benign_message_pool, N_BENIGN_SAMPLES
            )
        print(f"Sampled {len(sampled_benign_messages)} benign messages.")
    else:
        print("Skipping benign message sampling.")

    # Combine Samples and Save Output
    print("\nPhase 4: Combining msg samples and svaing")
    final_sampled_data = sampled_relevant_messages + sampled_benign_messages
    print(f"Total messages in final sample: {len(final_sampled_data)}")

    if not final_sampled_data:
        print("No messages were sampled. Cannot create output file.")
    else:
        df_final = pd.DataFrame(final_sampled_data)
        df_final["Cybercrime_Category"] = ""
        df_final["Annotation_Notes"] = ""
        output_columns = [
            "message_id",
            "channel_id",
            "date",
            "selection_reason",
            "previous_context",
            "text",
            "next_context",
            "Cybercrime_Category",
            "Annotation_Notes",
        ]
        df_final = df_final[[col for col in output_columns if col in df_final.columns]]
        df_final.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
        print(
            f"Successfully saved {len(df_final)} sampled messages to {OUTPUT_CSV_FILE}"
        )

    print("\nData sampling process finished.")
