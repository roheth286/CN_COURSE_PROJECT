import os
import sys
import glob
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from forecast.rollout import RecursiveForecaster, ForecastResult
from features.sequence_dataset import (
    NetworkStateScaler,
    split_session_chronologically,
    STATE_FEATURE_NAMES,
    STATE_VECTOR_DIM
)
from data.preprocess import INDEX_TO_CLASS, CLASS_TO_INDEX


@dataclass
class AttackEpisode:
    episode_id: int
    session_name: str
    attack_category: str
    class_index: int
    start_window_idx: int
    end_window_idx: int
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    num_windows: int = 0
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.num_windows == 0:
            self.num_windows = self.end_window_idx - self.start_window_idx + 1
        if self.duration_seconds == 0.0:
            self.duration_seconds = self.num_windows * 30.0


@dataclass
class EpisodeLeadTimeResult:
    episode_id: int
    session_name: str
    attack_category: str
    start_window_idx: int
    end_window_idx: int
    start_timestamp: Optional[str]
    detected: bool
    alert_window_idx: Optional[int]
    alert_timestamp: Optional[str]
    forecast_step_k: Optional[int]
    lead_time_seconds: float
    lead_time_windows: int
    detection_type: str
    threat_confidence: float
    predicted_category: str
    category_match: bool


@dataclass
class LeadTimeSummary:
    total_episodes: int
    detected_episodes: int
    early_warning_episodes: int
    onset_episodes: int
    delayed_episodes: int
    missed_episodes: int
    detection_rate: float
    early_warning_rate: float
    mean_lead_time_seconds: float
    median_lead_time_seconds: float
    max_lead_time_seconds: float
    category_metrics: Dict[str, Dict[str, Any]]
    total_benign_windows: int
    false_alarms: int
    false_alarm_rate: float
    episode_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "detected_episodes": self.detected_episodes,
            "early_warning_episodes": self.early_warning_episodes,
            "onset_episodes": self.onset_episodes,
            "delayed_episodes": self.delayed_episodes,
            "missed_episodes": self.missed_episodes,
            "detection_rate": self.detection_rate,
            "early_warning_rate": self.early_warning_rate,
            "mean_lead_time_seconds": self.mean_lead_time_seconds,
            "median_lead_time_seconds": self.median_lead_time_seconds,
            "max_lead_time_seconds": self.max_lead_time_seconds,
            "total_benign_windows": self.total_benign_windows,
            "false_alarms": self.false_alarms,
            "false_alarm_rate": self.false_alarm_rate,
            "category_metrics": self.category_metrics,
            "episode_details": self.episode_details,
        }


def identify_attack_episodes(
    dataframe: pd.DataFrame,
    session_name: str = "session",
    min_episode_len: int = 1
) -> List[AttackEpisode]:
    if len(dataframe) == 0 or "is_attack" not in dataframe.columns:
        return []

    is_attack_vals = dataframe["is_attack"].values
    categories = (
        dataframe["attack_category"].values
        if "attack_category" in dataframe.columns
        else ["BENIGN"] * len(dataframe)
    )
    class_indices = (
        dataframe["class_index"].values
        if "class_index" in dataframe.columns
        else [0] * len(dataframe)
    )
    timestamps = (
        dataframe["window_start"].astype(str).tolist()
        if "window_start" in dataframe.columns
        else [None] * len(dataframe)
    )
    end_timestamps = (
        dataframe["window_end"].astype(str).tolist()
        if "window_end" in dataframe.columns
        else [None] * len(dataframe)
    )

    episodes: List[AttackEpisode] = []
    in_episode = False
    ep_start = 0
    ep_cat = "BENIGN"
    ep_idx = 0
    current_episode_id = 1

    for i in range(len(dataframe)):
        is_att = (is_attack_vals[i] == 1)
        cat = categories[i]

        if is_att and not in_episode:
            in_episode = True
            ep_start = i
            ep_cat = cat
            ep_idx = int(class_indices[i])
        elif in_episode:
            if not is_att or (cat != ep_cat and cat != "BENIGN"):
                ep_end = i - 1
                if (ep_end - ep_start + 1) >= min_episode_len:
                    episodes.append(AttackEpisode(
                        episode_id=current_episode_id,
                        session_name=session_name,
                        attack_category=ep_cat,
                        class_index=ep_idx,
                        start_window_idx=ep_start,
                        end_window_idx=ep_end,
                        start_timestamp=timestamps[ep_start],
                        end_timestamp=end_timestamps[ep_end],
                    ))
                    current_episode_id += 1

                if is_att:
                    ep_start = i
                    ep_cat = cat
                    ep_idx = int(class_indices[i])
                else:
                    in_episode = False

    if in_episode:
        ep_end = len(dataframe) - 1
        if (ep_end - ep_start + 1) >= min_episode_len:
            episodes.append(AttackEpisode(
                episode_id=current_episode_id,
                session_name=session_name,
                attack_category=ep_cat,
                class_index=ep_idx,
                start_window_idx=ep_start,
                end_window_idx=ep_end,
                start_timestamp=timestamps[ep_start],
                end_timestamp=end_timestamps[ep_end],
            ))

    return episodes


class LeadTimeEngine:
    def __init__(
        self,
        forecaster: RecursiveForecaster,
        threat_threshold: float = 0.50,
        use_composite_threat: bool = True,
        window_seconds: int = 30
    ) -> None:
        self.forecaster = forecaster
        self.threat_threshold = threat_threshold
        self.use_composite_threat = use_composite_threat
        self.window_seconds = window_seconds

    def evaluate_stream(
        self,
        session_df: pd.DataFrame,
        session_name: str = "stream",
        history_len_m: int = 10,
        K: int = 5
    ) -> Tuple[List[EpisodeLeadTimeResult], int, int]:
        total_windows = len(session_df)
        if total_windows < history_len_m + 1:
            return [], 0, 0

        episodes = identify_attack_episodes(session_df, session_name=session_name)

        raw_features = session_df[STATE_FEATURE_NAMES].values
        if self.forecaster.scaler is not None:
            norm_features = self.forecaster.scaler.transform(raw_features)
        else:
            norm_features = raw_features.astype(np.float32)

        is_attack_arr = session_df["is_attack"].values
        timestamps = (
            session_df["window_start"].astype(str).tolist()
            if "window_start" in session_df.columns
            else [f"window_{i}" for i in range(total_windows)]
        )

        monitoring_steps = list(range(history_len_m - 1, total_windows - 1))
        batch_sequences = []
        for t in monitoring_steps:
            window_slice = norm_features[t - history_len_m + 1:t + 1]
            batch_sequences.append(window_slice)

        batch_tensor = np.array(batch_sequences, dtype=np.float32)

        forecast_res: ForecastResult = self.forecaster.rollout(
            sequences=batch_tensor,
            K=K,
            unscale=False,
            batch_size=128
        )

        if self.use_composite_threat:
            p_risk = forecast_res.risk_probabilities
            p_comp = forecast_res.composite_threat_probabilities
            threat_probs = np.maximum(p_risk, p_comp)
        else:
            threat_probs = forecast_res.risk_probabilities

        pred_categories = forecast_res.predicted_categories

        episode_results: Dict[int, EpisodeLeadTimeResult] = {}
        for ep in episodes:
            episode_results[ep.episode_id] = EpisodeLeadTimeResult(
                episode_id=ep.episode_id,
                session_name=ep.session_name,
                attack_category=ep.attack_category,
                start_window_idx=ep.start_window_idx,
                end_window_idx=ep.end_window_idx,
                start_timestamp=ep.start_timestamp,
                detected=False,
                alert_window_idx=None,
                alert_timestamp=None,
                forecast_step_k=None,
                lead_time_seconds=0.0,
                lead_time_windows=0,
                detection_type="Missed",
                threat_confidence=0.0,
                predicted_category="BENIGN",
                category_match=False
            )

        false_alarms = 0
        benign_monitoring_windows = 0

        for idx, t in enumerate(monitoring_steps):
            is_current_benign = (is_attack_arr[t] == 0)
            if is_current_benign:
                benign_monitoring_windows += 1

            step_threat_probs = threat_probs[idx]
            step_pred_cats = pred_categories[idx]

            alert_k_indices = np.where(step_threat_probs >= self.threat_threshold)[0]
            if len(alert_k_indices) == 0:
                alert_k_indices = np.where(step_pred_cats != 0)[0]

            if len(alert_k_indices) > 0:
                earliest_k_idx = int(alert_k_indices[0])
                k_step = earliest_k_idx + 1
                target_window = t + k_step
                confidence = float(step_threat_probs[earliest_k_idx])
                pred_cat_idx = int(step_pred_cats[earliest_k_idx])
                pred_cat_name = INDEX_TO_CLASS.get(pred_cat_idx, "Unknown")

                matched_episode: Optional[AttackEpisode] = None
                for ep in episodes:
                    if (ep.start_window_idx - K <= t <= ep.end_window_idx) and (
                        ep.start_window_idx <= target_window <= ep.end_window_idx + 2
                    ):
                        matched_episode = ep
                        break

                if matched_episode is not None:
                    res = episode_results[matched_episode.episode_id]
                    if not res.detected:
                        lead_windows = matched_episode.start_window_idx - t
                        lead_sec = float(lead_windows * self.window_seconds)

                        if lead_windows > 0:
                            det_type = "Proactive Early Warning"
                        elif lead_windows == 0:
                            det_type = "Onset Detection"
                        else:
                            det_type = "Delayed Detection"

                        res.detected = True
                        res.alert_window_idx = t
                        res.alert_timestamp = timestamps[t]
                        res.forecast_step_k = k_step
                        res.lead_time_windows = lead_windows
                        res.lead_time_seconds = lead_sec
                        res.detection_type = det_type
                        res.threat_confidence = confidence
                        res.predicted_category = pred_cat_name
                        res.category_match = (
                            pred_cat_name.lower() in matched_episode.attack_category.lower()
                            or matched_episode.attack_category.lower() in pred_cat_name.lower()
                        )
                else:
                    future_slice_end = min(t + K + 1, total_windows)
                    if np.sum(is_attack_arr[t + 1:future_slice_end]) == 0:
                        false_alarms += 1

        return list(episode_results.values()), false_alarms, benign_monitoring_windows

    def evaluate_all(
        self,
        state_vectors_dir: str = "Datasets/state_vectors",
        history_len_m: int = 10,
        K: int = 5,
        partition_mode: str = "all"
    ) -> LeadTimeSummary:
        search_path = os.path.join(PROJECT_ROOT, state_vectors_dir, "*.parquet")
        session_files = sorted(glob.glob(search_path))

        if len(session_files) == 0:
            raise FileNotFoundError(f"No state vector parquet files found in: {search_path}")

        all_results: List[EpisodeLeadTimeResult] = []
        total_false_alarms = 0
        total_benign_windows = 0

        for fpath in session_files:
            session_name = os.path.basename(fpath).replace("_window_30s.parquet", "")
            df = pd.read_parquet(fpath)

            if partition_mode == "test":
                _, _, test_df = split_session_chronologically(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
                eval_df = test_df.reset_index(drop=True)
            else:
                eval_df = df

            ep_results, fa, ben_win = self.evaluate_stream(
                session_df=eval_df,
                session_name=session_name,
                history_len_m=history_len_m,
                K=K
            )
            all_results.extend(ep_results)
            total_false_alarms += fa
            total_benign_windows += ben_win

        total_episodes = len(all_results)
        detected_episodes = [r for r in all_results if r.detected]
        early_warned = [r for r in detected_episodes if r.lead_time_seconds > 0]
        onset = [r for r in detected_episodes if r.lead_time_seconds == 0]
        delayed = [r for r in detected_episodes if r.lead_time_seconds < 0]
        missed = [r for r in all_results if not r.detected]

        detection_rate = len(detected_episodes) / max(total_episodes, 1)
        early_warning_rate = len(early_warned) / max(len(detected_episodes), 1)

        lead_times = [r.lead_time_seconds for r in detected_episodes]
        mean_lt = float(np.mean(lead_times)) if lead_times else 0.0
        median_lt = float(np.median(lead_times)) if lead_times else 0.0
        max_lt = float(np.max(lead_times)) if lead_times else 0.0

        fa_rate = total_false_alarms / max(total_benign_windows, 1)

        cat_map: Dict[str, List[EpisodeLeadTimeResult]] = {}
        for r in all_results:
            cat_map.setdefault(r.attack_category, []).append(r)

        category_metrics: Dict[str, Dict[str, Any]] = {}
        for cat_name, cat_res in cat_map.items():
            cat_detected = [r for r in cat_res if r.detected]
            cat_early = [r for r in cat_detected if r.lead_time_seconds > 0]
            cat_lts = [r.lead_time_seconds for r in cat_detected]

            category_metrics[cat_name] = {
                "total_episodes": len(cat_res),
                "detected": len(cat_detected),
                "early_warned": len(cat_early),
                "detection_rate": len(cat_detected) / max(len(cat_res), 1),
                "early_warning_rate": len(cat_early) / max(len(cat_detected), 1),
                "mean_lead_time_seconds": float(np.mean(cat_lts)) if cat_lts else 0.0,
                "median_lead_time_seconds": float(np.median(cat_lts)) if cat_lts else 0.0,
                "max_lead_time_seconds": float(np.max(cat_lts)) if cat_lts else 0.0,
            }

        episode_details = [asdict(r) for r in all_results]

        return LeadTimeSummary(
            total_episodes=total_episodes,
            detected_episodes=len(detected_episodes),
            early_warning_episodes=len(early_warned),
            onset_episodes=len(onset),
            delayed_episodes=len(delayed),
            missed_episodes=len(missed),
            detection_rate=float(detection_rate),
            early_warning_rate=float(early_warning_rate),
            mean_lead_time_seconds=mean_lt,
            median_lead_time_seconds=median_lt,
            max_lead_time_seconds=max_lt,
            category_metrics=category_metrics,
            total_benign_windows=total_benign_windows,
            false_alarms=total_false_alarms,
            false_alarm_rate=float(fa_rate),
            episode_details=episode_details
        )
