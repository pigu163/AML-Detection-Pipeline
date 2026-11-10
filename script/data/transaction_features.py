"""Build transaction features from historical transactions with PySpark."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Window sizes are in seconds for use with Spark's range-based windows.
WINDOWS = {"1h": 3600, "1d": 86400, "7d": 604800}


def account_events(transactions: DataFrame) -> DataFrame:
    """Create one outgoing and one incoming event per transaction."""
    # A self-transfer contributes to both directions for the same account.
    outgoing = transactions.selectExpr(
        "from_bank as bank",
        "from_account as account",
        "timestamp",
        "to_bank as other_bank",
        "to_account as other_account",
        "'out' as direction",
        "amount_paid as amount",
        "payment_currency as currency",
    )
    incoming = transactions.selectExpr(
        "to_bank as bank",
        "to_account as account",
        "timestamp",
        "from_bank as other_bank",
        "from_account as other_account",
        "'in' as direction",
        "amount_received as amount",
        "receiving_currency as currency",
    )
    return outgoing.unionByName(incoming)


def counterparty_history(events: DataFrame) -> DataFrame:
    """Count distinct counterparties in the previous day."""
    # Keep distinct account pairs and timestamps; repeated edges at one time
    # still represent a single counterparty.
    peers = (
        events.where(
            "bank is not null and account is not null and other_bank is not null and other_account is not null and timestamp is not null"
        )
        .select(
            "bank",
            "account",
            "other_bank",
            "other_account",
            "direction",
            F.unix_seconds("timestamp").alias("time"),
        )
        .distinct()
    )
    peer_window = Window.partitionBy(
        "bank", "account", "other_bank", "other_account", "direction"
    ).orderBy("time")
    peers = peers.withColumn("next_time", F.lead("time").over(peer_window))

    # A peer becomes active one second after its transaction and remains active
    # through the one-day boundary, unless its next transaction takes over.
    starts = peers.select(
        "bank",
        "account",
        "direction",
        (F.col("time") + 1).alias("time"),
        F.lit(1).alias("delta"),
    )
    ends = peers.select(
        "bank",
        "account",
        "direction",
        F.least(
            F.col("time") + 86401,
            F.coalesce(F.col("next_time") + 1, F.col("time") + 86401),
        ).alias("time"),
        F.lit(-1).alias("delta"),
    )
    # Add account transaction times so the cumulative count can be joined back.
    anchors = (
        events.select("bank", "account", F.unix_seconds("timestamp").alias("time"))
        .distinct()
        .select(
            "bank",
            "account",
            "time",
            F.lit(None).cast("string").alias("direction"),
            F.lit(0).alias("delta"),
        )
    )
    changes = (
        starts.unionByName(ends)
        .unionByName(anchors)
        .groupBy("bank", "account", "time")
        .agg(
            *[
                F.sum(
                    F.when(F.col("direction") == d, F.col("delta")).otherwise(0)
                ).alias(f"unique_{d}_counterparties_1d")
                for d in ("in", "out")
            ]
        )
    )
    # Summing interval starts (+1) and ends (-1) gives exact distinct counts.
    cumulative = (
        Window.partitionBy("bank", "account")
        .orderBy("time")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    counts = [f"unique_{d}_counterparties_1d" for d in ("in", "out")]
    return changes.select(
        "bank",
        "account",
        F.timestamp_seconds("time").alias("timestamp"),
        *[F.sum(c).over(cumulative).alias(c) for c in counts],
    )


def account_history(events: DataFrame) -> DataFrame:
    """Calculate account activity at each transaction time."""
    # Aggregate events at the same timestamp before computing historical windows.
    minutes = (
        events.groupBy("bank", "account", "timestamp")
        .agg(
            *[
                F.sum(F.when(F.col("direction") == d, 1).otherwise(0)).alias(f"{d}_n")
                for d in ("in", "out")
            ]
        )
        .withColumn("seconds", F.unix_seconds("timestamp"))
    )
    ordered = Window.partitionBy("bank", "account").orderBy("seconds")
    # Ending at -1 excludes all transactions at the current timestamp.
    history = ordered.rangeBetween(Window.unboundedPreceding, -1)
    expressions = [
        (F.col("seconds") - F.min("seconds").over(history)).alias(
            "seconds_since_first_activity"
        ),
        *[
            (
                F.col("seconds")
                - F.max(F.when(F.col(f"{d}_n") > 0, F.col("seconds"))).over(history)
            ).alias(f"seconds_since_last_{d}")
            for d in ("in", "out")
        ],
    ]
    for name, seconds in WINDOWS.items():
        # The lower bound is inclusive: [current time - window, current time).
        window = ordered.rangeBetween(-seconds, -1)
        expressions.extend(
            F.coalesce(F.sum(f"{d}_n").over(window), F.lit(0)).alias(
                f"{d}_count_{name}"
            )
            for d in ("in", "out")
        )
    result = minutes.select("bank", "account", "timestamp", *expressions)
    result = result.withColumn(
        "has_history", F.col("seconds_since_first_activity").isNotNull().cast("int")
    )
    return result.join(
        counterparty_history(events), ["bank", "account", "timestamp"], "left"
    )


def amount_history(transactions: DataFrame, events: DataFrame) -> DataFrame:
    """Aggregate incoming and outgoing amounts separately by currency."""
    # Add both transaction currencies as query points for each account, even
    # when an account has no earlier activity in one of those currencies.
    probes = (
        transactions.selectExpr(
            "from_bank as bank",
            "from_account as account",
            "timestamp",
            "payment_currency as paid",
            "receiving_currency as received",
        )
        .unionByName(
            transactions.selectExpr(
                "to_bank as bank",
                "to_account as account",
                "timestamp",
                "payment_currency as paid",
                "receiving_currency as received",
            )
        )
        .select(
            "bank",
            "account",
            "timestamp",
            F.explode(F.array("paid", "received")).alias("currency"),
        )
        .distinct()
    )
    probes = probes.withColumn("direction", F.lit(None).cast("string")).withColumn(
        "amount", F.lit(None).cast("double")
    )
    values = events.select(
        "bank", "account", "timestamp", "currency", "direction", "amount"
    )
    # Combine actual amounts and query points before the per-currency windows.
    minutes = (
        values.unionByName(probes)
        .groupBy("bank", "account", "timestamp", "currency")
        .agg(
            *[
                F.sum(F.when(F.col("direction") == d, F.col("amount"))).alias(
                    f"{d}_sum"
                )
                for d in ("in", "out")
            ],
            *[
                F.count(F.when(F.col("direction") == d, F.col("amount"))).alias(
                    f"{d}_count"
                )
                for d in ("in", "out")
            ],
            *[
                F.max(F.when(F.col("direction") == d, F.col("amount"))).alias(
                    f"{d}_max"
                )
                for d in ("in", "out")
            ],
        )
        .withColumn("seconds", F.unix_seconds("timestamp"))
    )
    ordered = Window.partitionBy("bank", "account", "currency").orderBy("seconds")
    expressions = []
    for name, seconds in WINDOWS.items():
        window = ordered.rangeBetween(-seconds, -1)
        for d in ("in", "out"):
            # Count non-null amounts for the mean; keep an empty mean as null.
            total = F.sum(f"{d}_sum").over(window)
            count = F.sum(f"{d}_count").over(window)
            expressions.extend(
                [
                    F.coalesce(total, F.lit(0.0)).alias(f"{d}_amount_sum_{name}"),
                    F.try_divide(total, count).alias(f"{d}_amount_mean_{name}"),
                    F.max(f"{d}_max").over(window).alias(f"{d}_amount_max_{name}"),
                ]
            )
    return minutes.select("bank", "account", "timestamp", "currency", *expressions)


def pair_history(transactions: DataFrame) -> DataFrame:
    """Calculate forward and reverse transaction-pair history."""
    keys = ["from_bank", "from_account", "to_bank", "to_account"]
    forward = transactions.select(*keys, "timestamp").withColumn("n", F.lit(1))
    # Zero-count reverse query points let us look up prior transfers both ways.
    reverse = (
        transactions.select(
            F.col("to_bank").alias("from_bank"),
            F.col("to_account").alias("from_account"),
            F.col("from_bank").alias("to_bank"),
            F.col("from_account").alias("to_account"),
            "timestamp",
        )
        .distinct()
        .withColumn("n", F.lit(0))
    )
    minutes = (
        forward.unionByName(reverse)
        .groupBy(*keys, "timestamp")
        .agg(F.sum("n").alias("n"))
        .withColumn("seconds", F.unix_seconds("timestamp"))
    )
    ordered = Window.partitionBy(*keys).orderBy("seconds")
    history = ordered.rangeBetween(Window.unboundedPreceding, -1)
    expressions = [
        (
            F.col("seconds")
            - F.max(F.when(F.col("n") > 0, F.col("seconds"))).over(history)
        ).alias("seconds_since")
    ]
    for name, seconds in WINDOWS.items():
        expressions.append(
            F.coalesce(
                F.sum("n").over(ordered.rangeBetween(-seconds, -1)), F.lit(0)
            ).alias(f"count_{name}")
        )
    return minutes.select(*keys, "timestamp", *expressions)


def transaction_features(transactions: DataFrame) -> DataFrame:
    """Return a lazy DataFrame; history always ends before the current minute."""
    # The input must contain the available history for one dataset only.
    events = account_events(transactions)
    history = account_history(events)
    amounts = amount_history(transactions, events)
    result = transactions.withColumns(
        {
            "is_cross_bank": (F.col("from_bank") != F.col("to_bank")).cast("int"),
            "is_cross_currency": (
                F.col("payment_currency") != F.col("receiving_currency")
            ).cast("int"),
            "hour_of_day": F.hour("timestamp"),
            "day_of_week": F.dayofweek("timestamp"),
        }
    )
    for side, prefix in (("src", "from"), ("dst", "to")):
        # Join the same account history to the sender and recipient separately.
        account_keys = [f"{prefix}_bank", f"{prefix}_account", "timestamp"]
        renamed = history.select(
            F.col("bank").alias(account_keys[0]),
            F.col("account").alias(account_keys[1]),
            "timestamp",
            *[F.col(c).alias(f"{side}_{c}") for c in history.columns[3:]],
        )
        result = result.join(renamed, account_keys, "left")
        for direction, currency in (
            ("in", "receiving_currency"),
            ("out", "payment_currency"),
        ):
            # Match each direction to the corresponding current currency.
            columns = [
                c for c in amounts.columns if c.startswith(f"{direction}_amount_")
            ]
            renamed = amounts.select(
                F.col("bank").alias(account_keys[0]),
                F.col("account").alias(account_keys[1]),
                "timestamp",
                F.col("currency").alias(currency),
                *[F.col(c).alias(f"{side}_{c}") for c in columns],
            )
            result = result.join(renamed, account_keys + [currency], "left")
    for name in WINDOWS:
        result = result.withColumn(
            f"src_paid_to_out_mean_ratio_{name}",
            F.try_divide("amount_paid", f"src_out_amount_mean_{name}"),
        )

    # Reversing the pair keys retrieves the opposite-direction history.
    pairs = pair_history(transactions)
    keys = ["from_bank", "from_account", "to_bank", "to_account", "timestamp"]
    forward = pairs.select(
        *keys,
        *[F.col(c).alias(f"pair_forward_{c}") for c in pairs.columns if c not in keys],
    )
    reverse = pairs.select(
        F.col("to_bank").alias("from_bank"),
        F.col("to_account").alias("from_account"),
        F.col("from_bank").alias("to_bank"),
        F.col("from_account").alias("to_account"),
        "timestamp",
        *[F.col(c).alias(f"pair_reverse_{c}") for c in pairs.columns if c not in keys],
    )
    result = result.join(forward, keys, "left").join(reverse, keys, "left")
    return (
        result.withColumn(
            "pair_is_first_forward",
            F.when(
                F.col("pair_forward_count_1h").isNotNull(),
                F.col("pair_forward_seconds_since").isNull().cast("int"),
            ),
        )
        .withColumnRenamed("pair_forward_seconds_since", "pair_seconds_since_forward")
        .withColumnRenamed("pair_reverse_seconds_since", "pair_seconds_since_reverse")
    )
