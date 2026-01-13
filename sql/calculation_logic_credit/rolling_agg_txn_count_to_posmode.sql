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

cte_base_join AS (
    SELECT
        Transaction_Serial_No,
        PANNumber,
        Transaction_Datetime,
        COALESCE(SAFE_CAST(CAST(POSMode AS INT) AS STRING), "__missing__") AS POSMode
    FROM cte_base
    LEFT JOIN Transaction_Summary_Calculation_Fraud AS tscf
        ON tscf.Transaction_Serial_No = cte_base.Transaction_Serial_No
),

cte_final AS (
    SELECT 
        A.Transaction_Serial_No,
        A.POSMode,
        A.CountryCode
        A.Transaction_Datetime,

        -- 15-minute window
        (
            SELECT COUNT(*) 
            FROM cte_base_join AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.POSMode = A.POSMode AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(MINUTE, -15, A.Transaction_Datetime)
        ) AS TxnCountToPOSMode_L15min,

        -- 30-day window
        (
            SELECT COUNT(*) 
            FROM cte_base_join AS B
            WHERE 
                B.PANNumber = A.PANNumber AND
                B.POSMode = A.POSMode AND
                B.Transaction_Datetime < A.Transaction_Datetime AND
                B.Transaction_Datetime > DATEADD(DAY, -30, A.Transaction_Datetime)
        ) AS TxnCountToPOSMode_L30D,

    FROM cte_base AS A
)

SELECT
    Transaction_Serial_No,
    PANNumber,
    TxnCountToPOSMode_L15min,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        TxnCountToPOSMode_L30D / 
        NULLIF(TxnCountToPOSMode_L15min, 0),
        -999      
    ) AS RatioTxnCountToPOSMode_L30DL15min
FROM cte_final