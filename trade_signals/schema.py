"""?? JSON ????"""

SIGNAL_SCHEMA = {
    "date": "2026-07-07",
    "generated_at": "2026-07-07 17:00:00",
    "signals": [
        {
            "stock": "000001",
            "action": "buy",
            "shares": 1000,
            "price": None,
            "price_type": "market",
        }
    ]
}

SIGNAL_EXAMPLE = {
    "date": "2026-07-07",
    "generated_at": "2026-07-07 17:00:00",
    "signals": [
        {"stock": "000001", "action": "buy",  "shares": 1000, "price": None,  "price_type": "market"},
        {"stock": "600519", "action": "sell", "shares": 200,  "price": 150.0, "price_type": "limit"},
        {"stock": "300750", "action": "hold", "shares": 0},
    ],
}
