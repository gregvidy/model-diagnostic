WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
    FROM C06_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
    FROM C10_Channel
),

cte_with_lag AS (
    SELECT
        *,
        LAG(Transaction_Datetime) OVER(
            PARTITION BY Debit_No
            ORDER BY Transaction_Datetime
        ) AS Prev_Transaction_Datetime
    FROM cte_base
),

SELECT
    Transaction_Serial_No,
    Debit_No,

    -- Time difference in mintues between current and previous txn
    COALESCE(
        DATEDIFF(MINUTE, Prev_Transaction_Datetime, Transaction_Datetime), 0
     ) AS TxnTimeDifference
FROM cte_with_lag