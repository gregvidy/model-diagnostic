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

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.PANNumber,
        A.Transaction_Datetime,

        -- 15-minute window
        (
            SELECT COUNT(*) 
            FROM cte_base AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS TxnCount_L15min,

        -- 30-day window
        (
            SELECT COUNT(*) 
            FROM cte_base AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS TxnCount_L30D,

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    PANNumber,
    TxnCount_L15min,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        TxnCount_L30D / 
        NULLIF(TxnCount_L15min, 0),
        0        
    ) AS RatioTxnCount_L30DL15min
FROM cte_final