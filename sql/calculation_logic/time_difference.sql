WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        Transaction_Datetime,
    FROM C03_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        Transaction_Datetime,
    FROM C09_Channel
),

cte_with_lag AS (
    SELECT
        *,
        LAG(Transaction_Datetime) OVER(
            PARTITION BY PANNumber
            ORDER BY Transaction_Datetime
        ) AS Prev_Transaction_Datetime
    FROM cte_base
),

SELECT
    Transaction_Serial_No,
    PANNumber,

    -- Time difference in mintues between current and previous txn
    COALESCE(
        DATEDIFF(MINUTE, Prev_Transaction_Datetime, Transaction_Datetime), 0
     ) AS TxnTimeDifference
FROM cte_with_lag