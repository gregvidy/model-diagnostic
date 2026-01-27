WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C03_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C09_Channel
),

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.PANNumber,
        A.MCC,
        A.Transaction_Datetime,

        -- Txn COUNT in the 15-minute and 30-days windows
        (
            SELECT COUNT(*)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS TxnCountToMCC_L15min,

        (
            SELECT COUNT(*)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS TxnCountToMCC_L30D,

        -- SUM Amount 15-minute and 30-days windows
        (
            SELECT SUM(B.Transaction_Amount)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS SumAmtToMCC_L15min,

        (
            SELECT SUM(B.Transaction_Amount)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS SumAmtToMCC_L30D,

        -- max Amount 30-day window
        (
            SELECT MAX(B.Transaction_Amount)
            FROM cte_base AS B
            WHERE 
                B.MCC = A.MCC AND
                B.PANNumber = A.PANNumber AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS MaxAmtToMCC_L30D,

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    PANNumber,
    MCC,
    Transaction_Datetime,
    MaxAmtToMCC_L30D,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        SumAmtToMCC_L30D / 
        NULLIF(SumAmtToMCC_L15min, 0),
        -999      
    ) AS RatioSumAmtToMCC_L30DL15min,
    ISNULL(
        TxnCountToMCC_L30D / 
        NULLIF(TxnCountToMCC_L15min, 0),
        -999    
    ) AS RatioTxnCountToMCC_L30DL15min,
FROM cte_final