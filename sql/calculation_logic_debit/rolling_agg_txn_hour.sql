WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
        DATEPART(HOUR, Transaction_Datetime) AS TxnHour
    FROM C06_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
        DATEPART(HOUR, Transaction_Datetime) AS TxnHour
    FROM C10_Channel
),

SELECT
    A.Transaction_Serial_No,
    A.Debit_No,

    -- Rolling avg of TxnHour over last 15 minutes
    (
        SELECT AVG(CAST(B.TxnHour AS FLOAT))
        FROM cte_base AS B
        WHERE
            B.Debit_No = A.Debit_No AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
    ) AS AvgTxnHour_L15min,

    -- Rolling avg of TxnHour over last 30 days
    (
        SELECT AVG(CAST(B.TxnHour AS FLOAT))
        FROM cte_base AS B
        WHERE
            B.Debit_No = A.Debit_No AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
    ) AS AvgTxnHour_L30D,

FROM cte_base AS A