WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
        COALESCE(Transaction_Amount, 0) AS Transaction_Amount,
    FROM C06_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
        COALESCE(Transaction_Amount, 0) AS Transaction_Amount,
    FROM C10_Channel
),

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.Debit_No,
        A.Transaction_Datetime,
        A.Transaction_Amount,

        -- SUM, AVG, MAX, 30-day window
        (
            SELECT SUM(A.Transaction_Amount) 
            FROM cte_base AS B
            WHERE 
                B.Debit_No = A.Debit_No AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS SumAmt_L30D,
        (
            SELECT AVG(A.Transaction_Amount) 
            FROM cte_base AS B
            WHERE 
                B.Debit_No = A.Debit_No AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS AvgAmt_L30D,
        (
            SELECT MAX(A.Transaction_Amount) 
            FROM cte_base AS B
            WHERE 
                B.Debit_No = A.Debit_No AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS MaxAmt_L30D,

        -- COUNT txn serial number 15-minutes and 30-day window
        (
            SELECT COUNT(*) 
            FROM cte_base AS B
            WHERE 
                B.Debit_No = A.Debit_No AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS TxnCount_L15min,
        (
            SELECT COUNT(*) 
            FROM cte_base AS B
            WHERE 
                B.Debit_No = A.Debit_No AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS TxnCount_L30D,

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    Debit_No,
    MaxAmt_L30D,
    AvgAmt_L30D,
    SumAmt_L30D,
    TxnCount_L30D,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        TxnCount_L30D / 
        NULLIF(TxnCount_L15min, 0),
        0        
    ) AS RatioTxnCount_L30DL15min
FROM cte_final