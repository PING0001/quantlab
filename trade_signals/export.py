"""???/??????? JSON ????"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def export_signals(
    signals: list[dict],
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    payload = {
        "date": today,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals": signals,
    }

    filename = f"signals_{today}.json"
    filepath = out_dir / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("?????: %s (%d ?)", filepath, len(signals))
        return filepath
    except OSError as e:
        logger.error("??????: %s", e)
        return None
