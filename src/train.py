from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd
from fpgrowth_py import fpgrowth

# Caminho padrão do modelo
DEFAULT_OUTPUT_PATH = "/mnt/model/model.pkl"

# Hiperparâmetros padrão
DEFAULT_MIN_SUPPORT = 0.05
DEFAULT_MIN_CONFIDENCE = 0.4

# Diretório onde ficam os CSVs
DEFAULT_DATASET_DIR = os.getenv("DATASET_DIR", "./datasets")

# Lista de CSVs
DEFAULT_DATASET_PATHS = [
    os.path.join(DEFAULT_DATASET_DIR, "2023_spotify_ds1.csv"),
    os.path.join(DEFAULT_DATASET_DIR, "2023_spotify_ds2.csv"),
]


@dataclass
class TrainingConfig:
    dataset_paths: List[str]
    output_path: str = DEFAULT_OUTPUT_PATH
    min_support: float = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_rules: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Treina o modelo de recomendação (FP-Growth).")

    parser.add_argument(
        "--output",
        dest="output_path",
        default=os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT_PATH),
        help="Onde o modelo treinado será salvo (padrão: /mnt/model/model.pkl).",
    )
    parser.add_argument(
        "--min-support",
        dest="min_support",
        type=float,
        default=float(os.getenv("MIN_SUPPORT", DEFAULT_MIN_SUPPORT)),
        help=f"Suporte mínimo relativo (padrão: {DEFAULT_MIN_SUPPORT}).",
    )
    parser.add_argument(
        "--min-confidence",
        dest="min_confidence",
        type=float,
        default=float(os.getenv("MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)),
        help=f"Confiança mínima (padrão: {DEFAULT_MIN_CONFIDENCE}).",
    )
    parser.add_argument(
        "--max-rules",
        dest="max_rules",
        type=int,
        default=None,
        help="Limite máximo de regras (opcional). Se não for passado, usa todas.",
    )

    args = parser.parse_args(argv)

    dataset_paths = DEFAULT_DATASET_PATHS

    return TrainingConfig(
        dataset_paths=dataset_paths,
        output_path=args.output_path,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_rules=args.max_rules,
    )


def _sanitize_track(track: str | float) -> str | None:
    if not isinstance(track, str):
        return None
    cleaned = track.strip()
    return cleaned or None


def load_transactions_single(dataset_path: str) -> List[List[str]]:
    print(f"[INFO] Lendo dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado em: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if "pid" not in df.columns or "track_name" not in df.columns:
        raise ValueError("Dataset deve conter as colunas 'pid' e 'track_name'.")

    cleaned = df["track_name"].map(_sanitize_track)
    df = df.assign(track_name=cleaned).dropna(subset=["track_name"])

    grouped = df.groupby("pid")["track_name"].apply(list)
    transactions = grouped.tolist()

    print(f"[INFO] {len(transactions)} playlists carregadas de {dataset_path}")
    return transactions


def load_transactions(dataset_paths: List[str]) -> List[List[str]]:
    all_transactions: List[List[str]] = []
    for path in dataset_paths:
        tx = load_transactions_single(path)
        all_transactions.extend(tx)

    print(f"[INFO] Total de playlists (ds1 + ds2): {len(all_transactions)}")
    return all_transactions


def normalise_rule(raw_rule) -> dict | None:
    antecedent: Iterable[str] | None = None
    consequent: Iterable[str] | None = None
    support: float | None = None
    confidence: float | None = None

    if isinstance(raw_rule, dict):
        antecedent = raw_rule.get("antecedent")
        consequent = raw_rule.get("consequent")
        support = raw_rule.get("support")
        confidence = raw_rule.get("confidence")
    elif isinstance(raw_rule, (list, tuple)) and len(raw_rule) >= 2:
        antecedent = raw_rule[0]
        consequent = raw_rule[1]
        numeric_tail = [value for value in raw_rule[2:] if isinstance(value, (int, float))]
        if numeric_tail:
            confidence = float(numeric_tail[-1])
            if len(numeric_tail) > 1:
                support = float(numeric_tail[0])
    else:
        return None

    def _ensure_iterable(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    antecedent_list = [item.strip() for item in _ensure_iterable(antecedent) if item and item.strip()]
    consequent_list = [item.strip() for item in _ensure_iterable(consequent) if item and item.strip()]

    if not antecedent_list or not consequent_list:
        return None

    return {
        "antecedent": antecedent_list,
        "consequent": consequent_list,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "support": float(support) if support is not None else None,
    }


def train_rules(transactions: List[List[str]], config: TrainingConfig) -> list[dict]:
    print(
        f"[INFO] Treinando FP-Growth (min_support={config.min_support}, "
        f"min_confidence={config.min_confidence})"
    )
    frequent_itemsets, raw_rules = fpgrowth(
        transactions,
        minSupRatio=config.min_support,
        minConf=config.min_confidence,
    )

    print(f"[INFO] Frequent itemsets encontrados: {len(frequent_itemsets)}")
    print(f"[INFO] Regras brutas encontradas: {len(raw_rules)}")

    rules = [normalise_rule(rule) for rule in raw_rules]
    rules = [rule for rule in rules if rule is not None]

    if config.max_rules is not None and len(rules) > config.max_rules:
        rules.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        rules = rules[: config.max_rules]

    print(f"[INFO] Regras finais após filtragem/limite: {len(rules)}")
    return rules


def persist_model(rules: list[dict], config: TrainingConfig) -> None:
    model = {
        "created_at": time.time(),
        "rules": rules,
        "params": {
            "min_support": config.min_support,
            "min_confidence": config.min_confidence,
            "max_rules": config.max_rules,
            "dataset_paths": config.dataset_paths,
        },
    }

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "wb") as file:
        pickle.dump(model, file)

    summary_path = os.path.splitext(config.output_path)[0] + ".json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "rule_count": len(rules),
                **model["params"],
            },
            summary_file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Modelo salvo em {config.output_path}")
    print(f"[OK] Resumo salvo em {summary_path}")


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)

    print("[INFO] Datasets usados para treinamento:")
    for path in config.dataset_paths:
        print("   -", path)

    transactions = load_transactions(config.dataset_paths)
    rules = train_rules(transactions, config)
    persist_model(rules, config)


if __name__ == "__main__":
    main()